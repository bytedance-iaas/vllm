# Batch Invariance

!!! note
    Batch invariance is currently in beta. Some features are still under active development.
    Track progress and planned improvements at <https://github.com/vllm-project/vllm/issues/27433>

This document shows how to enable batch invariance in vLLM. Batch invariance ensures that the output of a model is deterministic and independent of the batch size or the order of requests in a batch.

## Motivation

Batch invariance is crucial for several use cases:

- **Framework debugging**: Deterministic outputs make it easier to debug issues in the inference framework, as the same input will always produce the same output regardless of batching.
- **Model debugging**: Helps identify issues in model implementations by ensuring consistent behavior across different batch configurations.
- **Reinforcement Learning (RL)**: RL training often requires deterministic rollouts for reproducibility and stable training.
- **Large-scale inference systems**: Systems that use vLLM as a component benefit from deterministic behavior for testing, validation, and consistency guarantees.

## Hardware Requirements

Batch invariance is supported on the following platforms:

- NVIDIA GPUs with compute capability 8.0 or higher.
- AMD Instinct GPUs (gfx90a, gfx942, gfx950).
- Intel XPUs with Triton support.

### Attention Backend Selection for XPU

On XPU, Triton Attention backend is required for batch invariance.
Select this backend using Qwen/Qwen3-1.7B model as an example:

```python
llm = LLM(
    model="Qwen/Qwen3-1.7B",
    attention_config={"backend": "TRITON_ATTN"},
)
```

Or via the CLI:

```bash
VLLM_BATCH_INVARIANT=1 vllm serve Qwen/Qwen3-1.7B \
    --attention-config.backend TRITON_ATTN
```

## Enabling Batch Invariance

### Fixed sparse FP8 KV reduction on Hopper

For `FLASHMLA_SPARSE` with `fp8_ds_mla` on SM90, set
`VLLM_FLASHMLA_SPARSE_SINGLE_KV_SPLIT=1` to keep each request's entire
selected KV in one reduction. The default is `0`, which retains FlashMLA's
query-size-dependent split plan. The option preserves the FP8 cache and
native kernel; smaller query batches may lose split-KV parallelism.

This option controls the attention reduction only. Exact whole-model output
also depends on projections, indexer selection, routing and collectives, and
must be validated with the intended model and topology. It operates
independently of `VLLM_BATCH_INVARIANT`. Other GPU families and KV formats
are rejected when the option is enabled.

### Global batch invariance

Batch invariance can be enabled by setting the `VLLM_BATCH_INVARIANT` environment variable to `1`:

```bash
export VLLM_BATCH_INVARIANT=1
```

### Online Inference (Server Mode)

To start a vLLM server with batch invariance enabled:

```bash
VLLM_BATCH_INVARIANT=1 vllm serve meta-llama/Llama-3.1-8B-Instruct
```

Then use the OpenAI-compatible client:

```python
from openai import OpenAI

client = OpenAI(
    api_key="EMPTY",
    base_url="http://localhost:8000/v1",
)

# These requests will produce deterministic outputs
# regardless of batch size or order
response = client.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    prompt="The future of AI is",
    max_tokens=100,
    temperature=0.7,
    seed=42,
)

print(response.choices[0].text)
```

### Offline Inference

For offline batch inference with batch invariance:

```python
import os
os.environ["VLLM_BATCH_INVARIANT"] = "1"

from vllm import LLM, SamplingParams

prompts = [
    "The future of AI is",
    "Machine learning enables",
    "Deep learning models can",
]

sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.95,
    max_tokens=100,
    seed=42,
)

llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    tensor_parallel_size=1,
)

# Outputs will be deterministic regardless of batch size
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}")
    print(f"Generated: {generated_text!r}\n")
```

## Tested Models

Batch invariance has been tested and verified on the following models:

- **DeepSeek series**: `deepseek-ai/DeepSeek-V3`, `deepseek-ai/DeepSeek-V3-0324`, `deepseek-ai/DeepSeek-R1`, `deepseek-ai/DeepSeek-V3.1`
- **Qwen3 (Dense)**: `Qwen/Qwen3-1.7B`, `Qwen/Qwen3-8B`, `Qwen/Qwen3-4B-AWQ`, `Qwen/Qwen3-8B-AWQ`
- **Qwen3-VL (Vision-Language)**: `Qwen/Qwen3-VL-2B-Instruct`, `Qwen/Qwen3-VL-4B-Instruct` (single image and video inputs)
- **Qwen3 (MoE)**: `Qwen/Qwen3-30B-A3B`, `Qwen/Qwen3-Next-80B-A3B-Instruct`, `Qwen/Qwen3-30B-A3B-Thinking-2507-FP8`
- **Qwen2.5**: `Qwen/Qwen2.5-0.5B-Instruct`, `Qwen/Qwen2.5-1.5B-Instruct`, `Qwen/Qwen2.5-3B-Instruct`, `Qwen/Qwen2.5-7B-Instruct`, `Qwen/Qwen2.5-14B-Instruct`, `Qwen/Qwen2.5-32B-Instruct`, `Qwen/Qwen2.5-Coder-7B-Instruct`
- **Llama 3**: Llama3.1 and 3.2 series, `meta-llama/Llama-3.2-1B-Instruct`, `meta-llama/Llama-3.2-3B-Instruct` for example
- **GPT-OSS**: `openai/gpt-oss-20b`, `openai/gpt-oss-120b`
- **Mistral**: `mistralai/Mistral-7B-v0.3`
- **GLM-4 series**: `zai-org/glm-4-9b-chat`
- **Phi series**: `microsoft/Phi-3.5-mini-instruct`, `microsoft/phi-4`
- **Granite 3.1 (MoE)**: `ibm-granite/granite-3.1-1b-a400m-instruct`, `ibm-granite/granite-3.1-3b-a800m-instruct`
- **Granite 3.1 (Dense)**: `ibm-granite/granite-3.1-2b-instruct`, `ibm-granite/granite-3.1-8b-instruct`
- **EXAONE 4.0 series**: `LGAI-EXAONE/EXAONE-4.0-1.2B`, `LGAI-EXAONE/EXAONE-4.0.1-32B`, `LGAI-EXAONE/EXAONE-4.0-32B`
- **OLMo 2**: `allenai/OLMo-2-0425-1B-Instruct`
- **ERNIE 4.5**: `baidu/ERNIE-4.5-0.3B-PT`
- **SmolLM2**: `HuggingFaceTB/SmolLM2-1.7B-Instruct`
- **Gemma 2**: `google/gemma-2-2b-it`
- **PLaMo3**: `pfnet/plamo-3-nict-2b-base`
- **Granite MoE SWA**: `ibm-granite/granite-swash-3b-a600m` (MoE + sliding-window attention; FLASH_ATTN and TRITON_ATTN backends; FLEX_ATTENTION and decode-vs-prefill test excluded due to per-layer SWA KV window differences)
- **Granite SWA**: `ibm-granite/granite-swash-2b` (sliding-window attention; FLASH_ATTN and TRITON_ATTN backends)
- **Qwen3-0.6B**: `Qwen/Qwen3-0.6B`

Other models may also work, but these have been explicitly validated. If you encounter issues with a specific model, please report them on the [GitHub issue tracker](https://github.com/vllm-project/vllm/issues/new/choose).

## Implementation Details

### Deterministic sparse-indexer top-k

`VLLM_DETERMINISTIC_SPARSE_TOPK=1` enables canonical top-k selection independently
of batch invariance. Sparse indexers keep all candidates above the cutoff score,
then the lowest-index candidates tied at the cutoff, and return the selected
indices in ascending order. This also runs when `VLLM_BATCH_INVARIANT=1`.

The shared row-relative prefill and decode paths are covered. Fused SM120
prefill selection and approximate BF16 DeepSelect ranking are disabled under
this flag so the canonical pass can use the exact scores. DCP top-k merging and XPU
absolute-column prefill are outside its coverage. The flag keeps the existing
attention and MoE backends; it does not make every model operator batch invariant.

### Deterministic single-group MoE routing

`VLLM_DETERMINISTIC_GROUPED_TOPK=1` keeps CUDA single-group routing with
256 through 1023 experts on its native block kernel. Large routing batches
are processed in chunks of at most 1024 tokens, matching the weights used for
short batches. The default native block/warp transition can change unbiased
weights even when the selected expert IDs are identical.

This switch is independent of full batch invariance and also applies under
`VLLM_BATCH_INVARIANT=1`. It preserves the attention, quantization, MoE dispatch
and speculative-decoding backends. Other model operations still need their own
invariance validation; the switch alone does not guarantee model-wide equality.

### Deterministic MoE gate projection

`VLLM_DETERMINISTIC_MOE_GATE=1` independently makes unbiased SM90 gates with
bf16 input and weight, fp32 output, and dimensions `[6144, 256]` use the existing
persistent GEMM. This avoids changes in fp32 logits caused by token-count-based
kernel selection and split-K reduction. The output remains fp32, including
when `GateLinear.set_out_dtype` enables fp32 output after construction.

The switch defaults to off. Biased, quantized, other-dtype, other-dimension and
other-device gates retain their existing paths. This is separate from grouped
top-k: a stable top-k reduction also needs stable input logits. FP8 model
weights, KV cache formats, attention, expert dispatch and speculative decoding
keep their configurations. Model-wide invariance requires model validation.

### GLM-5.3 Prefill operator selection on SM90

`VLLM_GLM53_PREFILL_DETERMINISTIC_OPS=1` uses the existing fixed-order linear
kernel for the `(160, 6144)` BF16 indexer projection. For BF16 RMSNorm with
hidden size 6,144 and fewer than 256 rows, it uses the native large-batch
reduction by padding only the normalization inputs. The backbone completes the
TP all-reduce before residual addition and normalization on that path.

The switch defaults to off. Enable it only on the Prefill process of the
SM90 GLM-5.3 deployment. For FP8 sparse MLA, configure the attention route
independently with `attention_config.sparse_mla_force_mqa`. Verify generation
against the deployment's cache precision, attention backend, and topology.

`VLLM_FLASHMLA_SPARSE_SINGLE_KV_SPLIT=1` selects one KV split per request for
SM90 FP8 sparse FlashMLA with `fp8_ds_mla` KV. This keeps the attention reduction
partition fixed as the query shape changes. The switch defaults to off.

When batch invariance is enabled, vLLM:

1. Uses deterministic kernel implementations for attention and other operations
2. Ensures consistent numerical behavior across different batch sizes
3. Disables certain optimizations that may introduce non-determinism
4. Under tensor parallelism, keeps custom all-reduce on with a fixed reduction order (the 1-stage kernel is pinned, and large inputs are reduced in fixed-size chunks), and disables FlashInfer, AITER and QuickReduce all-reduce
5. On CUDA devices with tuned matmul table entries for the model's bf16 unquantized forward linear layers (Ada, Hopper, Blackwell),
   runs without `torch.compile` using breakable CUDA graphs so tile configs follow the runtime batch size; set `VLLM_USE_BREAKABLE_CUDAGRAPH=0` to opt out.
6. Splits the LoRA shrink reduction into 8 fixed K chunks that are summed in a fixed order instead of with atomics

!!! warning
    Batch invariance under tensor parallelism is not yet supported for all-reduces whose size isn't a multiple of 16 bytes, for example a hidden size that isn't a multiple of 8 in fp16/bf16. Such tensors can switch between NCCL and custom all-reduce depending on batch size. All validated models meet this requirement.

!!! note
    Enabling batch invariance may impact performance compared to the default non-deterministic mode. This trade-off is intentional to guarantee reproducibility.

## Future Improvements

The batch invariance feature is under active development. Planned improvements include:

- Support for additional GPU architectures
- Expanded model coverage
- Performance optimizations
- Additional testing and validation

For the latest status and to contribute ideas, see the [tracking issue](https://github.com/vllm-project/vllm/issues/27433).
