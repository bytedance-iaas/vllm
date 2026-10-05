# DeepSeek V4.1 Prefill Humming

This directory vendors the exact Humming runtime used by the qualified
DeepSeek V4.1 Flash Prefill configuration.

- Source: `https://github.com/inclusionAI/humming.git`
- Commit: `3632a5052b6e42011415ee1390cba6b614567059`
- Image path: `/opt/humming-dsv41-prefill-3632a505`
- Consumer: DeepSeek V4.1 Flash Prefill through `PYTHONPATH`

Decode intentionally keeps using `/opt/humming-0.1.12`. Do not replace that
path when updating this Prefill-only copy.

The retained stack uses grouped-contiguous Humming with W13 INT8 dynamic
GS128/FP32 input scales and W2 BF16 input. Its fixed-RPS Mean TTFT gains were
27.49% at 8K, 39.28% at 16K, 32.57% at 32K, and 12.47% at 128K.
