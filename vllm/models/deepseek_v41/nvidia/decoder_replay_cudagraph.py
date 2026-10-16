# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CUDA graphs of the decoder replay layers on a step's replay batch."""

from collections.abc import Callable
from copy import copy
from dataclasses import replace

import torch

from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphWrapper
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.config.kernel import NATIVE_MEGA_MOE_BACKENDS
from vllm.forward_context import (
    BatchDescriptor,
    ForwardContext,
    get_forward_context,
    override_forward_context,
)
from vllm.logger import init_logger
from vllm.models.deepseek_v41.decoder_replay_layers import DecoderReplayLayers
from vllm.v1.worker.gpu.cudagraph_utils import (
    BatchExecutionDescriptor,
    CudaGraphManager,
)
from vllm.v1.worker.gpu.input_batch import InputBuffers

logger = init_logger(__name__)


class DecoderReplayFrontCaptureDisabled(RuntimeError):
    """A coordinated pre-capture failure that may preserve the suffix graph."""


class DecoderReplayFrontCudaGraphManager(CudaGraphManager):
    """One breakable graph for the 224-row mixed L0-L19 prefix."""

    NUM_TOKENS = 224

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        run_front: Callable[..., tuple[torch.Tensor, ...]],
        *,
        use_inputs_embeds: bool,
        inputs_embeds_size: int,
        lookback_depth: int,
        num_aux_outputs: int,
        aux_hidden_size: int,
    ) -> None:
        compilation = copy(vllm_config.compilation_config)
        compilation.cudagraph_capture_sizes = [self.NUM_TOKENS]
        compilation.max_cudagraph_capture_size = self.NUM_TOKENS
        front_config = copy(vllm_config)
        front_config.compilation_config = compilation
        super().__init__(front_config, device, CUDAGraphMode.PIECEWISE, 1)
        self.breakable_cg_runner: BreakableCUDAGraphWrapper = BreakableCUDAGraphWrapper(
            run_front,
            front_config,
            runtime_mode=CUDAGraphMode.PIECEWISE,
        )
        self.input_buffers = InputBuffers(self.max_num_reqs, self.NUM_TOKENS, device)
        dtype = vllm_config.model_config.dtype
        assert isinstance(dtype, torch.dtype)
        self.inputs_embeds = (
            torch.zeros(
                self.NUM_TOKENS,
                inputs_embeds_size,
                dtype=dtype,
                device=device,
            )
            if use_inputs_embeds
            else None
        )
        self.lookback_token_ids = (
            torch.full(
                (self.max_num_reqs, lookback_depth),
                -1,
                dtype=torch.int32,
                device=device,
            )
            if lookback_depth > 0
            else None
        )
        self._runtime_context: ForwardContext | None = None
        self._runtime_num_tokens = 0
        self._num_aux_outputs = num_aux_outputs
        self._aux_hidden_size = aux_hidden_size
        self._dtype = dtype
        self._aux_output_buffers: list[torch.Tensor] = []

    def _inputs(
        self,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        return (
            self.input_buffers.input_ids,
            self.input_buffers.positions,
            self.inputs_embeds,
            self.lookback_token_ids,
        )

    def capture_front_graphs(
        self,
        prepare: Callable[[BatchExecutionDescriptor], ForwardContext],
        agree_ready: Callable[[bool], bool],
    ) -> None:
        """Capture the single prefix graph after ``prepare`` builds its metadata."""

        def create_forward_fn(desc: BatchExecutionDescriptor, warmup: bool):
            del warmup
            context = None
            local_ready = True
            try:
                context = prepare(desc)
            except Exception:
                local_ready = False
                logger.exception("DeepSeek V4.1 replay front preparation failed")
            if not agree_ready(local_ready):
                raise DecoderReplayFrontCaptureDisabled(
                    "decoder replay front preparation failed on a DP rank"
                )
            assert context is not None
            inputs = self._inputs()

            def forward_fn(cg_mode: CUDAGraphMode) -> None:
                with override_forward_context(
                    replace(context, cudagraph_runtime_mode=cg_mode)
                ):
                    outputs = self.breakable_cg_runner(*inputs)
                self._protect_aux_outputs(outputs)

            return forward_fn

        self.capture(create_forward_fn, "Capturing decoder replay front CUDA graph")
        if not agree_ready(self.has_graph()):
            raise DecoderReplayFrontCaptureDisabled(
                "decoder replay front capture produced no graph on a DP rank"
            )

    def has_graph(self) -> bool:
        entry = self.breakable_cg_runner.entries.get(
            BatchDescriptor(num_tokens=self.NUM_TOKENS)
        )
        capture = entry.capture if entry is not None else None
        return capture is not None and capture.num_graphs >= 1

    def prepare_runtime(self, context: ForwardContext, num_tokens: int) -> None:
        if not 0 < num_tokens <= self.NUM_TOKENS:
            raise ValueError(f"invalid decoder replay front rows: {num_tokens}")
        if self._runtime_context is not None:
            raise RuntimeError("decoder replay front runtime context is already armed")
        self._runtime_context = context
        self._runtime_num_tokens = num_tokens

    def clear_runtime(self) -> None:
        self._runtime_context = None
        self._runtime_num_tokens = 0

    def prepare_capture_buffers(self) -> None:
        if not self._aux_output_buffers:
            self._aux_output_buffers = [
                torch.empty(
                    (self.NUM_TOKENS, self._aux_hidden_size),
                    dtype=self._dtype,
                    device=self.device,
                )
                for _ in range(self._num_aux_outputs)
            ]

    def _protect_aux_outputs(
        self,
        outputs: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, ...]:
        if (
            not isinstance(outputs, tuple)
            or len(outputs) < 6
            or not all(isinstance(output, torch.Tensor) for output in outputs)
        ):
            raise RuntimeError("decoder replay front graph returned invalid outputs")
        aux_outputs = outputs[6:]
        if len(self._aux_output_buffers) != len(aux_outputs):
            raise RuntimeError("decoder replay front auxiliary output count changed")
        for buffer, output in zip(self._aux_output_buffers, aux_outputs):
            if buffer.shape != output.shape or buffer.dtype != output.dtype:
                raise RuntimeError("decoder replay front auxiliary shape changed")
            buffer.copy_(output)
        return outputs[:6] + tuple(self._aux_output_buffers)

    def run(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None,
        lookback_token_ids: torch.Tensor | None,
    ) -> tuple[torch.Tensor, ...]:
        context = self._runtime_context
        num_tokens = self._runtime_num_tokens
        self.clear_runtime()
        if context is None:
            raise RuntimeError("decoder replay front graph was not prepared")

        buffers = self.input_buffers
        buffers.input_ids[:num_tokens].copy_(input_ids[:num_tokens])
        buffers.input_ids[num_tokens:].zero_()
        buffers.positions[:num_tokens].copy_(positions[:num_tokens])
        buffers.positions[num_tokens:].zero_()
        if self.inputs_embeds is None:
            if inputs_embeds is not None:
                raise RuntimeError("decoder replay front graph expected token ids")
        else:
            if inputs_embeds is None:
                raise RuntimeError("decoder replay front graph expected input embeds")
            self.inputs_embeds[:num_tokens].copy_(inputs_embeds[:num_tokens])
            self.inputs_embeds[num_tokens:].zero_()
        if self.lookback_token_ids is not None:
            if lookback_token_ids is None:
                raise RuntimeError("decoder replay front graph expected lookback ids")
            self.lookback_token_ids.copy_(lookback_token_ids)

        with override_forward_context(context):
            outputs = self.breakable_cg_runner(*self._inputs())
        outputs = self._protect_aux_outputs(outputs)
        logger.info_once(
            "DSV41_REPLAY_FRONT_GRAPH_RUN rows=%d descriptor=%d",
            num_tokens,
            self.NUM_TOKENS,
        )
        return tuple(output[:num_tokens] for output in outputs)


class DecoderReplayCudaGraphManager(CudaGraphManager):
    """Breakable PIECEWISE graphs of the replay layers keyed by the replay batch's
    padded size. They read the layer inputs from static buffers."""

    def __init__(self, vllm_config: VllmConfig, device, layers: DecoderReplayLayers):
        compilation = copy(vllm_config.compilation_config)
        scheduler = vllm_config.scheduler_config
        sizes = compilation.decoder_replay_cudagraph_capture_sizes
        if not sizes:
            bound = min(
                compilation.max_cudagraph_capture_size,
                scheduler.max_num_seqs * layers.window,
            )
            # Replay batches hold at least `window` rows: step by window, or by 1/8
            # of the power-of-two interval once wider (256 in [2048, 4096)).
            sizes, s = [bound], layers.window
            while s < bound:
                sizes.append(s)
                s += max(layers.window, 1 << max(s.bit_length() - 4, 0))
        sizes = sorted(set(sizes))
        if max(sizes) > scheduler.max_num_batched_tokens:
            raise ValueError(
                "decoder_replay_cudagraph_capture_sizes must not exceed "
                "max_num_batched_tokens"
            )
        compilation.cudagraph_capture_sizes = sizes
        compilation.max_cudagraph_capture_size = max(sizes)
        replay_config = copy(vllm_config)
        replay_config.compilation_config = compilation
        super().__init__(replay_config, device, CUDAGraphMode.PIECEWISE, 1)
        self.layers = layers
        self.breakable_cg_runner: BreakableCUDAGraphWrapper = BreakableCUDAGraphWrapper(
            layers.run_layers, replay_config
        )
        config, act = vllm_config.model_config.hf_config, vllm_config.model_config.dtype
        assert isinstance(act, torch.dtype)
        hc, h, f32 = config.hc_mult, config.hidden_size, torch.float32
        # DeepseekV4Model.forward casts input_ids to int64 for native MegaMoE only.
        native = vllm_config.kernel_config.moe_backend in NATIVE_MEGA_MOE_BACKENDS
        ids = torch.int64 if native else torch.int32
        # hidden_states, positions, input_ids, pre_mix, post_mix, res_mix, residual
        shapes = [(h,), (), (), (hc,), (hc, 1), (hc, hc), (hc, h)]
        dtypes = [act, torch.int64, ids, f32, f32, f32, act]
        n = compilation.max_cudagraph_capture_size
        self.input_buffers = [
            torch.zeros(n, *s, dtype=d, device=device) for s, d in zip(shapes, dtypes)
        ]
        self._captured_forward_contexts: dict[BatchDescriptor, ForwardContext] = {}

    def capture_replay_graphs(
        self, prepare: Callable[[BatchExecutionDescriptor], None]
    ) -> None:
        """Capture each size on the replay batch ``prepare`` sets on the layers."""

        def create_forward_fn(desc: BatchExecutionDescriptor, warmup: bool):
            prepare(desc)
            assert self.layers.replay_batch is not None
            context = self.layers.replay_batch.forward_context
            if not warmup:
                batch_descriptor = context.batch_descriptor
                assert batch_descriptor is not None
                self._captured_forward_contexts[batch_descriptor] = context
            inputs = [buf[: desc.num_tokens] for buf in self.input_buffers]

            def forward_fn(cg_mode: CUDAGraphMode) -> None:
                with override_forward_context(
                    replace(context, cudagraph_runtime_mode=cg_mode)
                ):
                    self.breakable_cg_runner(*inputs)

            return forward_fn

        self.capture(create_forward_fn, "Capturing decoder replay CUDA graphs")
        self.layers.replay_batch = None

    def captured_forward_context(self, descriptor: BatchDescriptor) -> ForwardContext:
        """Return the context whose tensor addresses are bound by this graph."""
        try:
            return self._captured_forward_contexts[descriptor]
        except KeyError as error:
            raise RuntimeError(
                f"decoder replay graph has no captured context for {descriptor}"
            ) from error

    def clear_captured_forward_contexts(self) -> None:
        """Release contexts belonging to discarded profiling graphs."""
        self._captured_forward_contexts.clear()

    def run(self, rows: torch.Tensor, inputs: tuple) -> tuple[torch.Tensor, ...]:
        """Replay the forward context's graph on ``rows`` of ``inputs``."""
        num_rows = rows.shape[0]
        for buf, t in zip(self.input_buffers, inputs):
            torch.index_select(t, 0, rows, out=buf[:num_rows])
        batch_descriptor = get_forward_context().batch_descriptor
        assert batch_descriptor is not None
        num_tokens = batch_descriptor.num_tokens
        outputs = self.breakable_cg_runner(
            *(buf[:num_tokens] for buf in self.input_buffers)
        )
        return tuple(out[:num_rows] for out in outputs)
