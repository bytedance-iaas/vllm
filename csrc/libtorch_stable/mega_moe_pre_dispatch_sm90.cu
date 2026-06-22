/*
 * DeepSeek V4 MegaMoE SM90 (Hopper) input staging.
 *
 * Self-contained CUDA port of SGLang's
 * sgl-kernel/.../deepseek_v4/mega_moe_pre_dispatch_sm90.cuh, rewritten against
 * vLLM's stable-ABI custom-op interface (no sgl_kernel/* dependencies).
 *
 * One launch performs all of:
 *   - per-128-channel FP8 E4M3 quantization of the BF16 hidden states, writing
 *     the raw FP32 activation scale (NOT packed UE8M0);
 *   - copy of each token's top-k routing row into the symmetric buffer,
 *     folding routed_scaling_factor into the weights;
 *   - padding fill of the trailing rows [num_tokens, padded_max) with topk
 *     ids = -1 and weights = 0.0 so the DeepGEMM MegaMoE kernel skips them.
 *
 * Grid is num_tokens + ceil(pad_slots / blockDim) so the launch size scales
 * with the real token count plus the padding amount, not with padded_max.
 *
 * Licensed under the Apache License, Version 2.0.
 */

#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/macros.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>

#include "core/registration.h"
#include "libtorch_stable/torch_utils.h"

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdlib>
#include <mutex>

namespace {

inline int getSMVersion() {
  auto* props = get_device_prop();
  return props->major * 10 + props->minor;
}

inline bool getEnvEnablePDL() {
  static std::once_flag flag;
  static bool enablePDL = false;
  std::call_once(flag, [&]() {
    if (getSMVersion() >= 90) {
      char const* env = std::getenv("VLLM_ENABLE_PDL");
      enablePDL = env && env[0] == '1' && env[1] == '\0';
    }
  });
  return enablePDL;
}

using bf16_t = __nv_bfloat16;
using fp8_t = __nv_fp8_e4m3;

constexpr float kFP8E4M3Max = 448.0f;
constexpr float kEps = 1e-10f;
constexpr uint32_t kGroupSize = 128;   // SM90 per-128-channel scale
constexpr uint32_t kVecElems = 8;      // 8 bf16 = 16B vector load per thread
constexpr uint32_t kThreadsPerGroup = kGroupSize / kVecElems;  // 16

__device__ __forceinline__ float warpGroupReduceMax(float val) {
  // Reduce within the kThreadsPerGroup-sized subgroup (16 lanes).
#pragma unroll
  for (int offset = kThreadsPerGroup / 2; offset > 0; offset >>= 1) {
    val = fmaxf(val, __shfl_xor_sync(0xffffffffu, val, offset, 32));
  }
  return val;
}

__device__ __forceinline__ float fp8Clip(float v) {
  return fmaxf(fminf(v, kFP8E4M3Max), -kFP8E4M3Max);
}

// blockDim.x == hidden / kVecElems (one thread per 8-elem vector of a token).
// Each consecutive group of kThreadsPerGroup threads covers one 128-channel
// quant group.
__global__ void __launch_bounds__(1024, 2) mega_moe_pre_dispatch_sm90_kernel(
    const bf16_t* __restrict__ x,            // [num_tokens, hidden]
    const int32_t* __restrict__ topk_idx,    // [num_tokens, top_k]
    const float* __restrict__ topk_weights,  // [num_tokens, top_k]
    fp8_t* __restrict__ buf_x,               // [padded_max, hidden]
    float* __restrict__ buf_x_sf,            // [padded_max, hidden/128]
    int64_t* __restrict__ buf_topk_idx,      // [padded_max, top_k]
    float* __restrict__ buf_topk_weights,    // [padded_max, top_k]
    uint32_t num_tokens, uint32_t padded_max, uint32_t hidden,
    uint32_t num_groups, uint32_t top_k, float routed_scaling_factor,
    bool use_pdl) {
  const uint32_t bid = blockIdx.x;
  const uint32_t tid = threadIdx.x;

#if (defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900)
  if (use_pdl) {
    asm volatile("griddepcontrol.wait;" ::: "memory");
  }
#endif

  if (bid < num_tokens) {
    // ---- Quantize path: one CTA per valid token ----
    const uint32_t token_id = bid;
    const uint64_t base = static_cast<uint64_t>(token_id) * hidden;

    // 16B vectorized load of 8 bf16 values.
    float vals[kVecElems];
    float local_max = 0.0f;
    const uint32_t elem0 = tid * kVecElems;
    if (elem0 < hidden) {
      const float4 raw = *reinterpret_cast<const float4*>(x + base + elem0);
      const bf16_t* rb = reinterpret_cast<const bf16_t*>(&raw);
#pragma unroll
      for (uint32_t i = 0; i < kVecElems; ++i) {
        vals[i] = __bfloat162float(rb[i]);
        local_max = fmaxf(local_max, fabsf(vals[i]));
      }
    }

    // Absmax across the kThreadsPerGroup threads covering one 128-group.
    local_max = warpGroupReduceMax(local_max);
    const float absmax = fmaxf(local_max, kEps);
    const float raw_scale = absmax / kFP8E4M3Max;
    const float inv_scale = 1.0f / raw_scale;

    if (elem0 < hidden) {
      fp8_t out[kVecElems];
#pragma unroll
      for (uint32_t i = 0; i < kVecElems; ++i) {
        out[i] = static_cast<fp8_t>(fp8Clip(vals[i] * inv_scale));
      }
      *reinterpret_cast<float2*>(buf_x + base + elem0) =
          *reinterpret_cast<float2*>(out);
    }

    // One thread per group writes its raw FP32 scale.
    const uint32_t group_id = tid / kThreadsPerGroup;
    const uint32_t within_group = tid % kThreadsPerGroup;
    if (within_group == 0 && group_id < num_groups) {
      buf_x_sf[static_cast<uint64_t>(token_id) * num_groups + group_id] =
          raw_scale;
    }

    // Copy this token's topk row, folding routed_scaling_factor.
    if (tid < top_k) {
      const uint64_t off = static_cast<uint64_t>(token_id) * top_k + tid;
      buf_topk_idx[off] = static_cast<int64_t>(topk_idx[off]);
      buf_topk_weights[off] = topk_weights[off] * routed_scaling_factor;
    }
  } else {
    // ---- Pad path: trailing blocks fill [num_tokens, padded_max) ----
    const uint32_t copy_bid = bid - num_tokens;
    const uint64_t pad_base = static_cast<uint64_t>(num_tokens) * top_k;
    const uint64_t slot =
        pad_base + static_cast<uint64_t>(copy_bid) * blockDim.x + tid;
    const uint64_t total_slots = static_cast<uint64_t>(padded_max) * top_k;
    if (slot < total_slots) {
      buf_topk_idx[slot] = -1;
      buf_topk_weights[slot] = 0.0f;
    }
  }

#if (defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900)
  if (use_pdl) {
    asm volatile("griddepcontrol.launch_dependents;" :::);
  }
#endif
}

}  // namespace

void mega_moe_pre_dispatch_sm90(
    torch::stable::Tensor const& x, torch::stable::Tensor const& topk_idx,
    torch::stable::Tensor const& topk_weights, torch::stable::Tensor& buf_x,
    torch::stable::Tensor& buf_x_sf, torch::stable::Tensor& buf_topk_idx,
    torch::stable::Tensor& buf_topk_weights, double routed_scaling_factor) {
  using torch::headeronly::ScalarType;

  STD_TORCH_CHECK(x.dim() == 2, "x must be 2D [num_tokens, hidden]");
  STD_TORCH_CHECK(topk_idx.dim() == 2 && topk_weights.dim() == 2,
                  "topk tensors must be 2D");
  STD_TORCH_CHECK(buf_x.dim() == 2 && buf_x_sf.dim() == 2 &&
                      buf_topk_idx.dim() == 2 && buf_topk_weights.dim() == 2,
                  "buffers must be 2D");

  const int64_t num_tokens = x.size(0);
  const int64_t hidden = x.size(1);
  const int64_t padded_max = buf_x.size(0);
  const int64_t top_k = topk_idx.size(1);
  const int64_t num_groups = hidden / static_cast<int64_t>(kGroupSize);

  STD_TORCH_CHECK(num_tokens <= padded_max,
                  "num_tokens must not exceed padded_max");
  STD_TORCH_CHECK(hidden % kGroupSize == 0,
                  "hidden must be a multiple of 128");
  STD_TORCH_CHECK(hidden % kVecElems == 0,
                  "hidden must be a multiple of 8 (16B bf16 loads)");
  STD_TORCH_CHECK(topk_weights.size(0) == num_tokens &&
                      topk_weights.size(1) == top_k,
                  "topk_weights shape must match topk_idx");
  STD_TORCH_CHECK(buf_x.size(1) == hidden, "buf_x hidden mismatch");
  STD_TORCH_CHECK(buf_x_sf.size(0) == padded_max &&
                      buf_x_sf.size(1) == num_groups,
                  "buf_x_sf must be [padded_max, hidden/128]");
  STD_TORCH_CHECK(buf_topk_idx.size(0) == padded_max &&
                      buf_topk_idx.size(1) == top_k,
                  "buf_topk_idx must be [padded_max, top_k]");
  STD_TORCH_CHECK(buf_topk_weights.size(0) == padded_max &&
                      buf_topk_weights.size(1) == top_k,
                  "buf_topk_weights must be [padded_max, top_k]");

  STD_TORCH_CHECK(x.scalar_type() == ScalarType::BFloat16,
                  "x must be bfloat16");
  STD_TORCH_CHECK(topk_idx.scalar_type() == ScalarType::Int,
                  "topk_idx must be int32");
  STD_TORCH_CHECK(topk_weights.scalar_type() == ScalarType::Float,
                  "topk_weights must be float32");
  STD_TORCH_CHECK(buf_x.scalar_type() == ScalarType::Float8_e4m3fn,
                  "buf_x must be float8_e4m3fn");
  STD_TORCH_CHECK(buf_x_sf.scalar_type() == ScalarType::Float,
                  "buf_x_sf must be float32");
  STD_TORCH_CHECK(buf_topk_idx.scalar_type() == ScalarType::Long,
                  "buf_topk_idx must be int64");
  STD_TORCH_CHECK(buf_topk_weights.scalar_type() == ScalarType::Float,
                  "buf_topk_weights must be float32");

  // Contiguity: the kernel uses linear/vectorized addressing.
  STD_TORCH_CHECK(x.stride(1) == 1 && buf_x.stride(1) == 1,
                  "x and buf_x must be row-major contiguous");
  STD_TORCH_CHECK(buf_x_sf.stride(1) == 1 && buf_topk_idx.stride(1) == 1 &&
                      buf_topk_weights.stride(1) == 1,
                  "scale/topk buffers must be row-major contiguous");

  STD_TORCH_CHECK(getSMVersion() >= 90, "required CUDA ARCH >= SM_90");

  const uint32_t num_threads = static_cast<uint32_t>(hidden / kVecElems);
  STD_TORCH_CHECK(num_threads <= 1024,
                  "hidden too large for single-block-per-token quant");
  STD_TORCH_CHECK(num_threads >= static_cast<uint32_t>(top_k),
                  "top_k must fit into one quant CTA");

  if (padded_max == 0) return;

  const int64_t pad_slots = (padded_max - num_tokens) * top_k;
  const uint32_t num_pad_blocks =
      pad_slots <= 0 ? 0u
                     : static_cast<uint32_t>((pad_slots + num_threads - 1) /
                                             num_threads);
  const uint32_t num_blocks =
      static_cast<uint32_t>(num_tokens) + num_pad_blocks;
  if (num_blocks == 0) return;

  auto stream = get_current_cuda_stream(x.get_device_index());

  cudaLaunchConfig_t config;
  config.gridDim = dim3(num_blocks);
  config.blockDim = dim3(num_threads);
  config.dynamicSmemBytes = 0;
  config.stream = stream;
  cudaLaunchAttribute attrs[1];
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  const bool use_pdl = getEnvEnablePDL();
  attrs[0].val.programmaticStreamSerializationAllowed = use_pdl;
  config.numAttrs = 1;
  config.attrs = attrs;

  cudaLaunchKernelEx(
      &config, mega_moe_pre_dispatch_sm90_kernel,
      reinterpret_cast<const bf16_t*>(x.data_ptr()),
      reinterpret_cast<const int32_t*>(topk_idx.data_ptr()),
      reinterpret_cast<const float*>(topk_weights.data_ptr()),
      reinterpret_cast<fp8_t*>(buf_x.mutable_data_ptr()),
      reinterpret_cast<float*>(buf_x_sf.mutable_data_ptr()),
      reinterpret_cast<int64_t*>(buf_topk_idx.mutable_data_ptr()),
      reinterpret_cast<float*>(buf_topk_weights.mutable_data_ptr()),
      static_cast<uint32_t>(num_tokens), static_cast<uint32_t>(padded_max),
      static_cast<uint32_t>(hidden), static_cast<uint32_t>(num_groups),
      static_cast<uint32_t>(top_k), static_cast<float>(routed_scaling_factor),
      use_pdl);
  STD_CUDA_CHECK(cudaGetLastError());
}

STABLE_TORCH_LIBRARY_IMPL(_C, CUDA, m) {
  m.impl("mega_moe_pre_dispatch_sm90", TORCH_BOX(&mega_moe_pre_dispatch_sm90));
}
