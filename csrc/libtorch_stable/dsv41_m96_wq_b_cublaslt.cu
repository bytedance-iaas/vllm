// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>

#include "core/registration.h"
#include "libtorch_stable/torch_utils.h"

#include <cublasLt.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace {

constexpr int64_t kM = 96;
constexpr int64_t kN = 32768;
constexpr int64_t kK = 1280;
constexpr size_t kWorkspaceBytes = 4;
constexpr size_t kCublasLtVersion = 130101;

void check_cublas(cublasStatus_t status, const char* operation) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    throw std::runtime_error(std::string(operation) + " failed with status " +
                             std::to_string(static_cast<int>(status)));
  }
}

class Dsv41M96WqBCublasLtPlan {
 public:
  Dsv41M96WqBCublasLtPlan() {
    int device = 0;
    cudaDeviceProp properties{};
    STD_TORCH_CHECK(cudaGetDevice(&device) == cudaSuccess,
                    "dsv41_m96_wq_b_cublaslt: cudaGetDevice failed");
    STD_TORCH_CHECK(cudaGetDeviceProperties(&properties, device) == cudaSuccess,
                    "dsv41_m96_wq_b_cublaslt: cudaGetDeviceProperties failed");
    STD_TORCH_CHECK(
        properties.major == 9 && properties.minor == 0 &&
            properties.multiProcessorCount == 78 &&
            std::strstr(properties.name, "H20") != nullptr,
        "dsv41_m96_wq_b_cublaslt: the pinned tactic requires NVIDIA H20");
    STD_TORCH_CHECK(
        cublasLtGetVersion() == kCublasLtVersion,
        "dsv41_m96_wq_b_cublaslt: the pinned tactic requires cuBLASLt ",
        kCublasLtVersion);

    check_cublas(cublasLtCreate(&handle_), "cublasLtCreate");
    check_cublas(
        cublasLtMatmulDescCreate(&operation_, CUBLAS_COMPUTE_32F, CUDA_R_32F),
        "cublasLtMatmulDescCreate");
    cublasOperation_t transpose_a = CUBLAS_OP_T;
    cublasOperation_t transpose_b = CUBLAS_OP_N;
    check_cublas(
        cublasLtMatmulDescSetAttribute(operation_, CUBLASLT_MATMUL_DESC_TRANSA,
                                       &transpose_a, sizeof(transpose_a)),
        "set TRANSA");
    check_cublas(
        cublasLtMatmulDescSetAttribute(operation_, CUBLASLT_MATMUL_DESC_TRANSB,
                                       &transpose_b, sizeof(transpose_b)),
        "set TRANSB");

    check_cublas(cublasLtMatrixLayoutCreate(&a_, CUDA_R_16BF, kK, kN, kK),
                 "create A layout");
    check_cublas(cublasLtMatrixLayoutCreate(&b_, CUDA_R_16BF, kK, kM, kK),
                 "create B layout");
    check_cublas(cublasLtMatrixLayoutCreate(&c_, CUDA_R_16BF, kN, kM, kN),
                 "create C layout");
    check_cublas(cublasLtMatrixLayoutCreate(&d_, CUDA_R_16BF, kN, kM, kN),
                 "create D layout");

    constexpr int kAlgorithmId = 66;
    check_cublas(cublasLtMatmulAlgoInit(handle_, CUBLAS_COMPUTE_32F, CUDA_R_32F,
                                        CUDA_R_16BF, CUDA_R_16BF, CUDA_R_16BF,
                                        CUDA_R_16BF, kAlgorithmId, &algorithm_),
                 "cublasLtMatmulAlgoInit");

    set_algorithm_attribute(CUBLASLT_ALGO_CONFIG_TILE_ID, uint32_t{35});
    set_algorithm_attribute(CUBLASLT_ALGO_CONFIG_SPLITK_NUM, int32_t{1});
    set_algorithm_attribute(CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME,
                            uint32_t{CUBLASLT_REDUCTION_SCHEME_NONE});
    set_algorithm_attribute(CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, uint32_t{0});
    set_algorithm_attribute(CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, uint32_t{1});
    set_algorithm_attribute(CUBLASLT_ALGO_CONFIG_STAGES_ID, uint32_t{35});
    set_algorithm_attribute(CUBLASLT_ALGO_CONFIG_INNER_SHAPE_ID, uint16_t{0});
    set_algorithm_attribute(CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID, uint16_t{3});

    cublasLtMatmulHeuristicResult_t result{};
    check_cublas(cublasLtMatmulAlgoCheck(handle_, operation_, a_, b_, c_, d_,
                                         &algorithm_, &result),
                 "cublasLtMatmulAlgoCheck");
    STD_TORCH_CHECK(result.state == CUBLAS_STATUS_SUCCESS &&
                        result.workspaceSize <= kWorkspaceBytes,
                    "dsv41_m96_wq_b_cublaslt: pinned tactic validation failed");
  }

  ~Dsv41M96WqBCublasLtPlan() {
    if (d_ != nullptr) {
      cublasLtMatrixLayoutDestroy(d_);
    }
    if (c_ != nullptr) {
      cublasLtMatrixLayoutDestroy(c_);
    }
    if (b_ != nullptr) {
      cublasLtMatrixLayoutDestroy(b_);
    }
    if (a_ != nullptr) {
      cublasLtMatrixLayoutDestroy(a_);
    }
    if (operation_ != nullptr) {
      cublasLtMatmulDescDestroy(operation_);
    }
    if (handle_ != nullptr) {
      cublasLtDestroy(handle_);
    }
  }

  void run(void* output, const void* input, const void* weight, void* workspace,
           cudaStream_t stream) {
    float alpha = 1.0F;
    float beta = 0.0F;
    check_cublas(cublasLtMatmul(handle_, operation_, &alpha, weight, a_, input,
                                b_, &beta, output, c_, output, d_, &algorithm_,
                                workspace, kWorkspaceBytes, stream),
                 "cublasLtMatmul");
  }

 private:
  template <typename T>
  void set_algorithm_attribute(cublasLtMatmulAlgoConfigAttributes_t attribute,
                               T value) {
    check_cublas(cublasLtMatmulAlgoConfigSetAttribute(&algorithm_, attribute,
                                                      &value, sizeof(value)),
                 "cublasLtMatmulAlgoConfigSetAttribute");
  }

  cublasLtHandle_t handle_ = nullptr;
  cublasLtMatmulDesc_t operation_ = nullptr;
  cublasLtMatrixLayout_t a_ = nullptr;
  cublasLtMatrixLayout_t b_ = nullptr;
  cublasLtMatrixLayout_t c_ = nullptr;
  cublasLtMatrixLayout_t d_ = nullptr;
  cublasLtMatmulAlgo_t algorithm_{};
};

Dsv41M96WqBCublasLtPlan& get_plan(int device) {
  static std::mutex mutex;
  static std::unordered_map<int, std::unique_ptr<Dsv41M96WqBCublasLtPlan>>
      plans;
  std::lock_guard<std::mutex> lock(mutex);
  auto& plan = plans[device];
  if (plan == nullptr) {
    plan = std::make_unique<Dsv41M96WqBCublasLtPlan>();
  }
  return *plan;
}

}  // namespace

bool dsv41_m96_wq_b_cublaslt_is_supported(
    const torch::stable::Tensor& device_anchor) {
  if (!device_anchor.is_cuda()) {
    return false;
  }
  try {
    const int device = device_anchor.get_device_index();
    const torch::stable::accelerator::DeviceGuard device_guard(device);
    get_plan(device);
    return true;
  } catch (...) {
    return false;
  }
}

void dsv41_m96_wq_b_cublaslt(torch::stable::Tensor& output,
                             const torch::stable::Tensor& input,
                             const torch::stable::Tensor& weight,
                             torch::stable::Tensor& workspace) {
  STD_TORCH_CHECK(output.is_cuda() && input.is_cuda() && weight.is_cuda() &&
                      workspace.is_cuda(),
                  "dsv41_m96_wq_b_cublaslt: all tensors must be CUDA tensors");
  STD_TORCH_CHECK(
      output.get_device_index() == input.get_device_index() &&
          output.get_device_index() == weight.get_device_index() &&
          output.get_device_index() == workspace.get_device_index(),
      "dsv41_m96_wq_b_cublaslt: all tensors must be on the same device");
  STD_TORCH_CHECK(output.is_contiguous() && input.is_contiguous() &&
                      weight.is_contiguous() && workspace.is_contiguous(),
                  "dsv41_m96_wq_b_cublaslt: all tensors must be contiguous");
  STD_TORCH_CHECK(
      output.scalar_type() == torch::headeronly::ScalarType::BFloat16 &&
          input.scalar_type() == torch::headeronly::ScalarType::BFloat16 &&
          weight.scalar_type() == torch::headeronly::ScalarType::BFloat16,
      "dsv41_m96_wq_b_cublaslt: input, weight, and output must be bfloat16");
  STD_TORCH_CHECK(
      workspace.scalar_type() == torch::headeronly::ScalarType::Byte &&
          workspace.numel() >= static_cast<int64_t>(kWorkspaceBytes),
      "dsv41_m96_wq_b_cublaslt: workspace must contain at least 4 uint8 "
      "elements");
  STD_TORCH_CHECK(
      input.dim() == 2 && input.size(0) == kM && input.size(1) == kK,
      "dsv41_m96_wq_b_cublaslt: input must have shape [96, 1280]");
  STD_TORCH_CHECK(
      weight.dim() == 2 && weight.size(0) == kN && weight.size(1) == kK,
      "dsv41_m96_wq_b_cublaslt: weight must have shape [32768, 1280]");
  STD_TORCH_CHECK(
      output.dim() == 2 && output.size(0) == kM && output.size(1) == kN,
      "dsv41_m96_wq_b_cublaslt: output must have shape [96, 32768]");

  const int device = input.get_device_index();
  const torch::stable::accelerator::DeviceGuard device_guard(device);
  auto stream = get_current_cuda_stream(device);
  get_plan(device).run(output.mutable_data_ptr(), input.data_ptr(),
                       weight.data_ptr(), workspace.mutable_data_ptr(), stream);
}

STABLE_TORCH_LIBRARY_IMPL(_C, CUDA, m) {
  m.impl("dsv41_m96_wq_b_cublaslt", TORCH_BOX(&dsv41_m96_wq_b_cublaslt));
  m.impl("dsv41_m96_wq_b_cublaslt_is_supported",
         TORCH_BOX(&dsv41_m96_wq_b_cublaslt_is_supported));
}
