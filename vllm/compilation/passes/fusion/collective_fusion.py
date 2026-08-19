# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import operator
from collections.abc import Callable
from contextlib import suppress

import torch
import torch._inductor.pattern_matcher as pm
import torch.distributed.distributed_c10d as c10d
import torch.fx as fx
from torch._inductor.pattern_matcher import PatternMatcherPass
from torch.distributed._symmetric_memory import enable_symm_mem_for_group

from vllm.config import CUDAGraphMode, VllmConfig
from vllm.config.utils import Range
from vllm.distributed import get_tp_group
from vllm.distributed.parallel_state import (
    get_tensor_model_parallel_world_size,
)
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.torch_utils import direct_register_custom_op

from ..inductor_pass import enable_fake_mode
from ..vllm_inductor_pass import (
    VllmFusionPatternMatcherPass,
    VllmInductorPass,
    VllmPatternMatcherPass,
    VllmPatternReplacement,
)

FP8_DTYPE = current_platform.fp8_dtype()

logger = init_logger(__name__)


def _flashinfer_scaled_mm_out(
    A: torch.Tensor,
    B: torch.Tensor,
    *,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    out: torch.Tensor,
    bias: torch.Tensor | None = None,
    scale_result: torch.Tensor | None = None,
    out_dtype: torch.dtype | None = None,
    use_fast_accum: bool = False,
) -> None:
    # Import lazily to avoid a circular import during module initialization
    # when docs or other tooling import the pass without FlashInfer.
    from vllm.utils.flashinfer import flashinfer_scaled_fp8_mm_out

    assert bias is None, "FlashInfer symm_mem adapter does not support bias"
    assert scale_result is None, (
        "FlashInfer symm_mem adapter does not support result scaling"
    )
    assert not use_fast_accum, (
        "FlashInfer symm_mem adapter does not support use_fast_accum"
    )
    assert A.ndim == 2 and B.ndim == 2 and out.ndim == 2, (
        "FlashInfer symm_mem adapter expects 2D inputs and output"
    )
    assert scale_a.numel() == 1 and scale_b.numel() == 1, (
        "FlashInfer symm_mem adapter only supports tensor-wise FP8 scales"
    )

    flashinfer_scaled_fp8_mm_out(
        A,
        B,
        scale_a,
        scale_b,
        out=out,
        out_dtype=out_dtype or out.dtype,
    )


def _flashinfer_fp4_mm_out(
    A: torch.Tensor,
    B: torch.Tensor,
    *,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    out: torch.Tensor,
    alpha: torch.Tensor,
    out_dtype: torch.dtype | None = None,
    use_8x4_sf_layout: bool = False,
    backend: str = "cutlass",
) -> None:
    from vllm.utils.flashinfer import flashinfer_scaled_fp4_mm_out

    assert A.ndim == 2 and B.ndim == 2 and out.ndim == 2, (
        "FlashInfer FP4 symm_mem adapter expects 2D inputs and output"
    )
    flashinfer_scaled_fp4_mm_out(
        A,
        B,
        scale_a,
        scale_b,
        alpha,
        out=out,
        out_dtype=out_dtype or out.dtype,
        use_8x4_sf_layout=use_8x4_sf_layout,
        backend=backend,
    )


def fused_flashinfer_scaled_matmul_reduce_scatter_fake(
    A: torch.Tensor,
    B: torch.Tensor,
    A_scale: torch.Tensor,
    B_scale: torch.Tensor,
    reduce_op: str,
    orig_scatter_dim: int,
    scatter_dim_after_maybe_reshape: int,
    group_name: str,
    output_shape: list[int],
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    world_size = c10d._resolve_process_group(group_name).size()
    result_shape = list(output_shape)
    result_shape[orig_scatter_dim] //= world_size
    return torch.empty(
        result_shape,
        dtype=out_dtype or torch.bfloat16,
        device=A.device,
    )


def fused_flashinfer_scaled_matmul_reduce_scatter(
    A: torch.Tensor,
    B: torch.Tensor,
    A_scale: torch.Tensor,
    B_scale: torch.Tensor,
    reduce_op: str,
    orig_scatter_dim: int,
    scatter_dim_after_maybe_reshape: int,
    group_name: str,
    output_shape: list[int],
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    assert orig_scatter_dim == 0 and scatter_dim_after_maybe_reshape == 0, (
        "FlashInfer symm_mem adapter currently only supports scatter_dim=0"
    )
    world_size = c10d._resolve_process_group(group_name).size()
    assert A.ndim == 2 and B.ndim == 2, "FlashInfer symm_mem adapter expects 2D inputs"
    assert A.is_contiguous(), "FlashInfer symm_mem adapter expects contiguous A"
    assert A_scale.numel() == 1 and B_scale.numel() == 1, (
        "FlashInfer symm_mem adapter only supports tensor-wise FP8 scales"
    )
    assert A.shape[0] % world_size == 0, (
        "FlashInfer symm_mem adapter expects M divisible by world size"
    )

    kwargs = {
        "scale_b": B_scale,
        "bias": None,
        "scale_result": None,
        "out_dtype": out_dtype,
        "use_fast_accum": False,
    }
    return torch.distributed._symmetric_memory._fused_scaled_matmul_reduce_scatter_impl(
        mm_out_op=_flashinfer_scaled_mm_out,
        A=A,
        B=B,
        A_scale=A_scale,
        kwargs=kwargs,
        out_dtype=out_dtype,
        reduce_op=reduce_op,
        orig_scatter_dim=orig_scatter_dim,
        scatter_dim_after_maybe_reshape=scatter_dim_after_maybe_reshape,
        group_name=group_name,
        output_shape=output_shape,
    )


def fused_all_gather_flashinfer_scaled_matmul_fake(
    A_shard: torch.Tensor,
    B: torch.Tensor,
    A_scale: torch.Tensor,
    B_scale: torch.Tensor,
    gather_dim: int,
    group_name: str,
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    world_size = c10d._resolve_process_group(group_name).size()
    output_shape = list(A_shard.shape)
    output_shape[gather_dim] *= world_size
    output_shape[-1] = B.shape[1]
    return torch.empty(
        output_shape,
        dtype=out_dtype or torch.bfloat16,
        device=A_shard.device,
    )


def fused_all_gather_flashinfer_scaled_matmul(
    A_shard: torch.Tensor,
    B: torch.Tensor,
    A_scale: torch.Tensor,
    B_scale: torch.Tensor,
    gather_dim: int,
    group_name: str,
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    assert gather_dim == 0, (
        "FlashInfer symm_mem adapter currently only supports gather_dim=0"
    )
    _, outputs = torch.distributed._symmetric_memory._fused_all_gather_matmul_impl(
        mm_out_op=_flashinfer_scaled_mm_out,
        A_shard=A_shard,
        Bs=[B],
        A_scale=A_scale,
        kwargs_list=[
            {
                "scale_b": B_scale,
                "bias": None,
                "scale_result": None,
                "out_dtype": out_dtype,
                "use_fast_accum": False,
            }
        ],
        out_dtypes=[out_dtype],
        gather_dim=gather_dim,
        group_name=group_name,
        return_A=False,
    )
    return outputs[0]


def fused_all_gather_flashinfer_fp4_matmul_fake(
    A_shard: torch.Tensor,
    B: torch.Tensor,
    A_scale_shard: torch.Tensor,
    B_scale: torch.Tensor,
    alpha: torch.Tensor,
    gather_dim: int,
    group_name: str,
    out_dtype: torch.dtype | None = None,
    view_a_scale_as_fp8: bool = False,
    use_8x4_sf_layout: bool = False,
    backend: str = "cutlass",
) -> torch.Tensor:
    world_size = c10d._resolve_process_group(group_name).size()
    output_shape = list(A_shard.shape)
    output_shape[gather_dim] *= world_size
    output_shape[-1] = B.shape[1]
    return torch.empty(
        output_shape,
        dtype=out_dtype or torch.bfloat16,
        device=A_shard.device,
    )


def fused_all_gather_flashinfer_fp4_matmul(
    A_shard: torch.Tensor,
    B: torch.Tensor,
    A_scale_shard: torch.Tensor,
    B_scale: torch.Tensor,
    alpha: torch.Tensor,
    gather_dim: int,
    group_name: str,
    out_dtype: torch.dtype | None = None,
    view_a_scale_as_fp8: bool = False,
    use_8x4_sf_layout: bool = False,
    backend: str = "cutlass",
) -> torch.Tensor:
    assert gather_dim == 0, (
        "FlashInfer FP4 symm_mem adapter currently only supports gather_dim=0"
    )
    assert A_shard.ndim == 2 and A_scale_shard.ndim == 2 and B.ndim == 2, (
        "FlashInfer FP4 symm_mem adapter expects 2D inputs"
    )
    if view_a_scale_as_fp8:
        A_scale_shard = A_scale_shard.view(torch.float8_e4m3fn)

    group = c10d._resolve_process_group(group_name)
    world_size = group.size()
    output = A_shard.new_empty(
        A_shard.shape[0] * world_size,
        B.shape[1],
        dtype=out_dtype or torch.bfloat16,
    )
    output_shards = output.chunk(world_size)

    A = A_shard.new_empty(A_shard.shape[0] * world_size, A_shard.shape[1])
    A_scale = A_scale_shard.new_empty(
        A_scale_shard.shape[0] * world_size,
        A_scale_shard.shape[1],
    )

    def fp4_shard_consumer(shards: list[torch.Tensor], rank: int) -> None:
        _flashinfer_fp4_mm_out(
            shards[0],
            B,
            scale_a=shards[1],
            scale_b=B_scale,
            alpha=alpha,
            out=output_shards[rank],
            out_dtype=out_dtype,
            use_8x4_sf_layout=use_8x4_sf_layout,
            backend=backend,
        )

    torch.distributed._symmetric_memory._pipelined_multi_all_gather_and_consume(
        [A_shard, A_scale_shard],
        fp4_shard_consumer,
        [A, A_scale],
        group_name,
        False,
    )
    return output


direct_register_custom_op(
    op_name="fused_flashinfer_scaled_matmul_reduce_scatter",
    op_func=fused_flashinfer_scaled_matmul_reduce_scatter,
    fake_impl=fused_flashinfer_scaled_matmul_reduce_scatter_fake,
)

direct_register_custom_op(
    op_name="fused_all_gather_flashinfer_scaled_matmul",
    op_func=fused_all_gather_flashinfer_scaled_matmul,
    fake_impl=fused_all_gather_flashinfer_scaled_matmul_fake,
)

direct_register_custom_op(
    op_name="fused_all_gather_flashinfer_fp4_matmul",
    op_func=fused_all_gather_flashinfer_fp4_matmul,
    fake_impl=fused_all_gather_flashinfer_fp4_matmul_fake,
)


class BasePattern:
    def __init__(self, dtype: torch.dtype, device: str | None) -> None:
        self.dtype = dtype
        self.device = device
        self.tp = get_tp_group()
        self.tp_size = get_tensor_model_parallel_world_size()


class GEMMReduceScatterPattern(BasePattern):
    def get_inputs(self) -> list[torch.Tensor]:
        mul = torch.empty([16, 4], device=self.device, dtype=self.dtype)
        mm_weight = torch.empty([4, 4], device=self.device, dtype=self.dtype)
        return [mul, mm_weight]

    def register(self, pm_pass: PatternMatcherPass) -> None:
        def pattern(mul: torch.Tensor, mm_weight: torch.Tensor) -> torch.Tensor:
            mm = torch.ops.aten.mm.default(mul, mm_weight)
            reduce_scatter = torch.ops.vllm.reduce_scatter.default(
                mm,
                dim=0,
                world_size=self.tp_size,
                group_name=self.tp.unique_name,
            )
            return reduce_scatter

        def replacement(mul: torch.Tensor, mm_weight: torch.Tensor) -> torch.Tensor:
            gemm_rs = torch.ops.symm_mem.fused_matmul_reduce_scatter(
                mul,
                mm_weight,
                "sum",
                scatter_dim=0,
                group_name=self.tp.device_group.group_name,
            )

            return gemm_rs

        pm.register_replacement(
            pattern, replacement, self.get_inputs(), pm.fwd_only, pm_pass
        )


class AllGatherGEMMPattern(BasePattern):
    def get_inputs(self) -> list[torch.Tensor]:
        x = torch.empty([4, 4], device=self.device, dtype=self.dtype)
        weight = torch.empty([4, 4], device=self.device, dtype=self.dtype)

        return [x, weight]

    def register(self, pm_pass: PatternMatcherPass) -> None:
        def pattern(
            x: torch.Tensor,
            weight: torch.Tensor,
        ) -> torch.Tensor:
            all_gather = torch.ops.vllm.all_gather.default(
                x,
                dim=0,
                world_size=self.tp_size,
                group_name=self.tp.unique_name,
            )

            return torch.ops.aten.mm.default(all_gather, weight)

        def replacement(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
            ag_output, mm_outputs = torch.ops.symm_mem.fused_all_gather_matmul(
                x,
                [weight],
                gather_dim=0,
                group_name=self.tp.device_group.group_name,
            )
            return mm_outputs

        pm.register_replacement(
            pattern, replacement, self.get_inputs(), pm.fwd_only, pm_pass
        )


class ScaledMMReduceScatterPattern(BasePattern):
    def get_inputs(self) -> list[torch.Tensor]:
        input = torch.empty([16, 16], device=self.device, dtype=FP8_DTYPE)
        mm_weight = (
            torch.empty([16, 16], device=self.device, dtype=FP8_DTYPE)
            .contiguous()
            .transpose(0, 1)
        )
        scale_a = torch.empty([16, 1], device=self.device, dtype=torch.float32)
        scale_b = torch.empty([1, 16], device=self.device, dtype=torch.float32)
        return [input, mm_weight, scale_a, scale_b]

    def register(self, pm_pass: PatternMatcherPass) -> None:
        def pattern(
            input: torch.Tensor,
            mat2: torch.Tensor,
            scale_a: torch.Tensor,
            scale_b: torch.Tensor,
        ) -> torch.Tensor:
            scaled_mm = torch.ops.aten._scaled_mm.default(
                input,
                mat2=mat2,
                scale_a=scale_a,
                scale_b=scale_b,
                bias=None,
                scale_result=None,
                out_dtype=self.dtype,
            )
            reduce_scatter = torch.ops.vllm.reduce_scatter.default(
                scaled_mm,
                dim=0,
                world_size=self.tp_size,
                group_name=self.tp.unique_name,
            )
            return reduce_scatter

        def replacement(
            input: torch.Tensor,
            mat2: torch.Tensor,
            scale_a: torch.Tensor,
            scale_b: torch.Tensor,
        ) -> torch.Tensor:
            # Calculate output shape: input @ mat2 with scatter_dim reduced
            output_shape = [*input.shape[:-1], mat2.shape[1]]
            scatter_dim = 0
            gemm_rs = torch.ops.vllm.patched_fused_scaled_matmul_reduce_scatter(
                input,
                mat2,
                scale_a,
                scale_b,
                "sum",
                scatter_dim,  # orig_scatter_dim
                scatter_dim,  # scatter_dim_after_maybe_reshape
                self.tp.device_group.group_name,
                output_shape,
                None,  # bias
                None,  # result_scale
                self.dtype,  # out_dtype
                False,  # use_fast_accum
            )

            return gemm_rs

        pm.register_replacement(
            pattern, replacement, self.get_inputs(), pm.fwd_only, pm_pass
        )


class AllGatherScaledMMPattern(BasePattern):
    def get_inputs(self) -> list[torch.Tensor]:
        x = torch.empty([8, 16], device=self.device, dtype=FP8_DTYPE)
        weight = (
            torch.empty([16, 16], device=self.device, dtype=FP8_DTYPE)
            .contiguous()
            .transpose(0, 1)
        )

        s1 = x.shape[0] * self.tp_size

        scale_a = torch.empty([s1, 1], device=self.device, dtype=torch.float32)
        scale_b = torch.empty([1, 16], device=self.device, dtype=torch.float32)

        return [x, weight, scale_a, scale_b]

    def register(self, pm_pass: PatternMatcherPass) -> None:
        def pattern(
            x: torch.Tensor,
            weight: torch.Tensor,
            scale_a: torch.Tensor,
            scale_b: torch.Tensor,
        ) -> torch.Tensor:
            all_gather = torch.ops.vllm.all_gather.default(
                x, dim=0, world_size=self.tp_size, group_name=self.tp.unique_name
            )

            return torch.ops.aten._scaled_mm.default(
                all_gather,
                mat2=weight,
                scale_a=scale_a,
                scale_b=scale_b,
                bias=None,
                scale_result=None,
                out_dtype=self.dtype,
            )

        def replacement(
            x: torch.Tensor,
            weight: torch.Tensor,
            scale_a: torch.Tensor,
            scale_b: torch.Tensor,
        ) -> torch.Tensor:
            ag_output, mm_outputs = torch.ops.symm_mem.fused_all_gather_scaled_matmul(  # noqa
                x,
                [weight],
                scale_a,
                [scale_b],
                gather_dim=0,
                biases=[None],
                result_scales=[None],
                out_dtypes=[self.dtype],
                use_fast_accum=[False],
                group_name=self.tp.device_group.group_name,
            )
            return mm_outputs

        pm.register_replacement(
            pattern, replacement, self.get_inputs(), pm.fwd_only, pm_pass
        )


class CutlassScaledMMReduceScatterPattern(BasePattern):
    def get_inputs(self) -> list[torch.Tensor]:
        input = torch.empty([16, 16], device=self.device, dtype=FP8_DTYPE)
        mm_weight = (
            torch.empty([16, 16], device=self.device, dtype=FP8_DTYPE)
            .contiguous()
            .transpose(0, 1)
        )
        scale_a = torch.empty([16, 1], device=self.device, dtype=torch.float32)
        scale_b = torch.empty([1, 16], device=self.device, dtype=torch.float32)

        cutlass_mm_output = torch.empty([16, 16], device=self.device, dtype=self.dtype)
        return [input, mm_weight, scale_a, scale_b, cutlass_mm_output]

    def register(self, pm_pass: PatternMatcherPass) -> None:
        def pattern(
            input: torch.Tensor,
            weight: torch.Tensor,
            scale_a: torch.Tensor,
            scale_b: torch.Tensor,
            cutlass_mm_output: torch.Tensor,
        ) -> torch.Tensor:
            cutlass_scaled_mm = torch.ops.higher_order.auto_functionalized(
                torch.ops._C.cutlass_scaled_mm.default,
                out=cutlass_mm_output,
                a=input,
                b=weight,
                a_scales=scale_a,
                b_scales=scale_b,
                bias=None,
            )

            reduce_scatter = torch.ops.vllm.reduce_scatter.default(
                cutlass_scaled_mm[1],
                dim=0,
                world_size=self.tp_size,
                group_name=self.tp.unique_name,
            )
            return reduce_scatter

        def replacement(
            input: torch.Tensor,
            mat2: torch.Tensor,
            scale_a: torch.Tensor,
            scale_b: torch.Tensor,
            cutlass_mm_output: torch.Tensor,
        ) -> torch.Tensor:
            # Calculate output shape: input @ mat2 with scatter_dim reduced
            output_shape = [*input.shape[:-1], mat2.shape[1]]
            scatter_dim = 0
            gemm_rs = torch.ops.vllm.patched_fused_scaled_matmul_reduce_scatter(
                input,
                mat2,
                scale_a,
                scale_b,
                "sum",
                scatter_dim,  # orig_scatter_dim
                scatter_dim,  # scatter_dim_after_maybe_reshape
                self.tp.device_group.group_name,
                output_shape,
                None,  # bias
                None,  # result_scale
                self.dtype,  # out_dtype
                False,  # use_fast_accum
            )

            return gemm_rs

        pm.register_replacement(
            pattern, replacement, self.get_inputs(), pm.fwd_only, pm_pass
        )


class AllGatherCutlassScaledMMPattern(BasePattern):
    def get_inputs(self) -> list[torch.Tensor]:
        x = torch.empty([8, 16], device=self.device, dtype=FP8_DTYPE)
        weight = (
            torch.empty([16, 16], device=self.device, dtype=FP8_DTYPE)
            .contiguous()
            .transpose(0, 1)
        )

        s1 = x.shape[0] * self.tp_size

        scale_a = torch.empty([s1, 1], device=self.device, dtype=torch.float32)
        scale_b = torch.empty([1, 16], device=self.device, dtype=torch.float32)

        s2 = weight.shape[1]
        output = torch.empty([s1, s2], device=self.device, dtype=self.dtype)

        return [x, weight, scale_a, scale_b, output]

    def register(self, pm_pass: PatternMatcherPass) -> None:
        def pattern(
            x: torch.Tensor,
            weight: torch.Tensor,
            scale_a: torch.Tensor,
            scale_b: torch.Tensor,
            output: torch.Tensor,
        ) -> torch.Tensor:
            all_gather = torch.ops.vllm.all_gather.default(
                x, dim=0, world_size=self.tp_size, group_name=self.tp.unique_name
            )

            cutlass_scaled_mm = torch.ops.higher_order.auto_functionalized(
                torch.ops._C.cutlass_scaled_mm.default,
                out=output,
                a=all_gather,
                b=weight,
                a_scales=scale_a,
                b_scales=scale_b,
                bias=None,
            )
            return cutlass_scaled_mm[1]

        def replacement(
            x: torch.Tensor,
            weight: torch.Tensor,
            scale_a: torch.Tensor,
            scale_b: torch.Tensor,
            output: torch.Tensor,
        ) -> torch.Tensor:
            ag_output, mm_outputs = torch.ops.symm_mem.fused_all_gather_scaled_matmul(  # noqa
                x,
                [weight],
                scale_a,
                [scale_b],
                gather_dim=0,
                biases=[None],
                result_scales=[None],
                out_dtypes=[self.dtype],
                use_fast_accum=[False],
                group_name=self.tp.device_group.group_name,
            )
            return mm_outputs

        pm.register_replacement(
            pattern, replacement, self.get_inputs(), pm.fwd_only, pm_pass
        )


class FlashInferBMMFP8ReduceScatterPattern(
    BasePattern, VllmPatternReplacement[..., torch.Tensor]
):
    def get_inputs(self) -> list[torch.Tensor]:
        a_2d = torch.empty([16, 16], device=self.device, dtype=FP8_DTYPE)
        b_2d = (
            torch.empty([16, 16], device=self.device, dtype=FP8_DTYPE)
            .contiguous()
            .transpose(0, 1)
        )
        a_scale = torch.empty([1], device=self.device, dtype=torch.float32)
        b_scale = torch.empty([1], device=self.device, dtype=torch.float32)
        return [a_2d, b_2d, a_scale, b_scale]

    @property
    def pattern(self) -> Callable[..., torch.Tensor]:
        def _pattern(
            a_2d: torch.Tensor,
            b_2d: torch.Tensor,
            a_scale: torch.Tensor,
            b_scale: torch.Tensor,
        ) -> torch.Tensor:
            bmm = torch.ops.vllm.bmm_fp8.default(
                torch.ops.aten.unsqueeze.default(a_2d, 0),
                torch.ops.aten.unsqueeze.default(b_2d, 0),
                a_scale,
                b_scale,
                self.dtype,
                "auto",
            )
            output = torch.ops.aten.reshape.default(bmm, list(bmm.shape[1:]))
            return torch.ops.vllm.reduce_scatter.default(
                output,
                dim=0,
                world_size=self.tp_size,
                group_name=self.tp.unique_name,
            )

        return _pattern

    @property
    def replacement(self) -> Callable[..., torch.Tensor]:
        def _replacement(
            a_2d: torch.Tensor,
            b_2d: torch.Tensor,
            a_scale: torch.Tensor,
            b_scale: torch.Tensor,
        ) -> torch.Tensor:
            return torch.ops.vllm.fused_flashinfer_scaled_matmul_reduce_scatter.default(
                a_2d,
                b_2d,
                a_scale,
                b_scale,
                "sum",
                0,
                0,
                self.tp.device_group.group_name,
                [a_2d.shape[0], b_2d.shape[1]],
                self.dtype,
            )

        return _replacement


class FlashInferAllGatherBMMFP8Pattern(
    BasePattern, VllmPatternReplacement[..., torch.Tensor]
):
    def get_inputs(self) -> list[torch.Tensor]:
        a_shard_2d = torch.empty([8, 16], device=self.device, dtype=FP8_DTYPE)
        b_2d = (
            torch.empty([16, 16], device=self.device, dtype=FP8_DTYPE)
            .contiguous()
            .transpose(0, 1)
        )
        a_scale = torch.empty([1], device=self.device, dtype=torch.float32)
        b_scale = torch.empty([1], device=self.device, dtype=torch.float32)
        return [a_shard_2d, b_2d, a_scale, b_scale]

    @property
    def pattern(self) -> Callable[..., torch.Tensor]:
        def _pattern(
            a_shard_2d: torch.Tensor,
            b_2d: torch.Tensor,
            a_scale: torch.Tensor,
            b_scale: torch.Tensor,
        ) -> torch.Tensor:
            all_gather = torch.ops.vllm.all_gather.default(
                a_shard_2d,
                dim=0,
                world_size=self.tp_size,
                group_name=self.tp.unique_name,
            )
            return torch.ops.vllm.bmm_fp8.default(
                torch.ops.aten.unsqueeze.default(all_gather, 0),
                torch.ops.aten.unsqueeze.default(b_2d, 0),
                a_scale,
                b_scale,
                self.dtype,
                "auto",
            )

        return _pattern

    @property
    def replacement(self) -> Callable[..., torch.Tensor]:
        def _replacement(
            a_shard_2d: torch.Tensor,
            b_2d: torch.Tensor,
            a_scale: torch.Tensor,
            b_scale: torch.Tensor,
        ) -> torch.Tensor:
            fused = torch.ops.vllm.fused_all_gather_flashinfer_scaled_matmul.default(
                a_shard_2d,
                b_2d,
                a_scale,
                b_scale,
                0,
                self.tp.device_group.group_name,
                self.dtype,
            )
            return torch.ops.aten.unsqueeze.default(fused, 0)

        return _replacement


class FlashInferAllGatherFP4Pattern(
    BasePattern, VllmPatternReplacement[..., torch.Tensor]
):
    def __init__(
        self,
        dtype: torch.dtype,
        device: str | None,
        backend: str,
        use_8x4_sf_layout: bool,
        a_scale_view: str,
    ) -> None:
        super().__init__(dtype, device)
        self.backend = backend
        self.use_8x4_sf_layout = use_8x4_sf_layout
        self.a_scale_view = a_scale_view

    def get_inputs(self) -> list[torch.Tensor]:
        a_shard_2d = torch.empty([8, 8], device=self.device, dtype=torch.uint8)
        b_2d = torch.empty([8, 16], device=self.device, dtype=torch.uint8)
        a_scale_shard = torch.empty([128, 4], device=self.device, dtype=torch.int32)
        b_scale = torch.empty([4, 128], device=self.device, dtype=torch.uint8)
        alpha = torch.empty([], device=self.device, dtype=torch.float32)
        return [
            a_shard_2d,
            b_2d,
            a_scale_shard,
            b_scale,
            alpha,
        ]

    @property
    def pattern(self) -> Callable[..., torch.Tensor]:
        def _pattern(
            a_shard_2d: torch.Tensor,
            b_2d: torch.Tensor,
            a_scale_shard: torch.Tensor,
            b_scale: torch.Tensor,
            alpha: torch.Tensor,
        ) -> torch.Tensor:
            all_gather_a = torch.ops.vllm.all_gather.default(
                a_shard_2d,
                dim=0,
                world_size=self.tp_size,
                group_name=self.tp.unique_name,
            )
            all_gather_a_scale = torch.ops.vllm.all_gather.default(
                a_scale_shard,
                dim=0,
                world_size=self.tp_size,
                group_name=self.tp.unique_name,
            )
            a_scale = all_gather_a_scale
            if self.a_scale_view in ("float8", "float8_uint8"):
                a_scale = torch.ops.aten.view.dtype(a_scale, torch.float8_e4m3fn)
            if self.a_scale_view in ("uint8", "float8_uint8"):
                a_scale = torch.ops.aten.view.dtype(a_scale, torch.uint8)
            return torch.ops.vllm.flashinfer_mm_fp4.default(
                all_gather_a,
                b_2d,
                a_scale,
                b_scale,
                alpha,
                self.dtype,
                self.use_8x4_sf_layout,
                self.backend,
            )

        return _pattern

    @property
    def replacement(self) -> Callable[..., torch.Tensor]:
        def _replacement(
            a_shard_2d: torch.Tensor,
            b_2d: torch.Tensor,
            a_scale_shard: torch.Tensor,
            b_scale: torch.Tensor,
            alpha: torch.Tensor,
        ) -> torch.Tensor:
            return torch.ops.vllm.fused_all_gather_flashinfer_fp4_matmul.default(
                a_shard_2d,
                b_2d,
                a_scale_shard,
                b_scale,
                alpha,
                0,
                self.tp.device_group.group_name,
                self.dtype,
                self.a_scale_view in ("float8", "float8_uint8"),
                self.use_8x4_sf_layout,
                self.backend,
            )

        return _replacement


class AsyncTPPass(VllmFusionPatternMatcherPass):
    @enable_fake_mode
    def __init__(self, config: VllmConfig) -> None:
        super().__init__(config, pass_name="async_tp_pass")

        self.w4a8_chunked_layers: dict[str, float] = {}
        self.w4a8_chunked_wildcard_scale: float | None = None
        if (
            config.compilation_config.cudagraph_mode == CUDAGraphMode.NONE
            and self.model_dtype == torch.bfloat16
            and get_tensor_model_parallel_world_size() == 4
        ):
            all_moe_layer_names = config.compilation_config.static_all_moe_layers
            all_moe_layers = set(all_moe_layer_names)
            materialized_boundaries: set[str] = set()
            for layer_name in all_moe_layers:
                layer = config.compilation_config.static_forward_context.get(layer_name)
                if layer is None:
                    continue
                supports = getattr(
                    layer,
                    "supports_w4a8_chunked_finalize_rs",
                    None,
                )
                if supports is not None and supports():
                    self.w4a8_chunked_layers[layer_name] = layer.routed_scaling_factor
                    continue
                is_boundary = getattr(
                    layer,
                    "is_w4a8_chunked_finalize_rs_materialized_boundary",
                    None,
                )
                if is_boundary is not None and is_boundary():
                    materialized_boundaries.add(layer_name)
            scales = set(self.w4a8_chunked_layers.values())
            if (
                all_moe_layers
                and len(all_moe_layers) == len(all_moe_layer_names)
                and set(self.w4a8_chunked_layers) | materialized_boundaries
                == all_moe_layers
                and scales == {1.0}
            ):
                self.w4a8_chunked_wildcard_scale = 1.0
            logger.debug(
                "Eligible W4A8 chunked finalize-RS layers: %s/%s "
                "(materialized boundaries: %s)",
                len(self.w4a8_chunked_layers),
                len(all_moe_layers),
                len(materialized_boundaries),
            )

        enable_symm_mem_for_group(get_tp_group().device_group.group_name)
        GEMMReduceScatterPattern(self.model_dtype, self.device).register(self.pm_pass)

        AllGatherGEMMPattern(self.model_dtype, self.device).register(self.pm_pass)

        # These fusions are enabled only for bfloat16 models because
        # `scaled_mm` or `cutlass_scaled_mm` with per-token (row-wise) scaling
        # only supports bfloat16 as the output dtype.
        if self.model_dtype == torch.bfloat16:
            ScaledMMReduceScatterPattern(self.model_dtype, self.device).register(
                self.pm_pass
            )
            AllGatherScaledMMPattern(self.model_dtype, self.device).register(
                self.pm_pass
            )
            if hasattr(torch.ops._C, "cutlass_scaled_mm"):
                CutlassScaledMMReduceScatterPattern(
                    self.model_dtype, self.device
                ).register(self.pm_pass)
                AllGatherCutlassScaledMMPattern(self.model_dtype, self.device).register(
                    self.pm_pass
                )
            with suppress(ImportError):
                import vllm.utils.flashinfer  # noqa: F401
            if hasattr(torch.ops.vllm, "bmm_fp8"):
                self.register(
                    FlashInferAllGatherBMMFP8Pattern(self.model_dtype, self.device)
                )
                self.register(
                    FlashInferBMMFP8ReduceScatterPattern(self.model_dtype, self.device)
                )
            if hasattr(torch.ops.vllm, "flashinfer_mm_fp4"):
                for backend in ("cutlass", "cudnn"):
                    for a_scale_view in ("float8_uint8", "uint8"):
                        self.register(
                            FlashInferAllGatherFP4Pattern(
                                self.model_dtype,
                                self.device,
                                backend,
                                use_8x4_sf_layout=False,
                                a_scale_view=a_scale_view,
                            )
                        )
                for use_8x4_sf_layout in (False, True):
                    for a_scale_view in ("float8",):
                        self.register(
                            FlashInferAllGatherFP4Pattern(
                                self.model_dtype,
                                self.device,
                                "trtllm",
                                use_8x4_sf_layout=use_8x4_sf_layout,
                                a_scale_view=a_scale_view,
                            )
                        )
                self.register(
                    FlashInferAllGatherFP4Pattern(
                        self.model_dtype,
                        self.device,
                        "cute-dsl",
                        use_8x4_sf_layout=False,
                        a_scale_view="float8",
                    )
                )
                # NVFP4 reduce-scatter does not need scale communication: FP4
                # scales are consumed by the local GEMM and only BF16 partial
                # outputs are reduced. Keep this PR scoped to the all-gather
                # path; reduce-scatter needs a dedicated FP4 producer rather
                # than the existing FP8-style helper.

        self.dump_patterns(config, self.pm_pass)

    @staticmethod
    def _resolve_hoisted_layer_name(layer_name: object) -> str | None:
        value = layer_name
        if isinstance(value, fx.Node):
            value = value.meta.get("val")
        real_obj = getattr(value, "real_obj", None)
        if real_obj is not None:
            value = real_obj
        resolved = getattr(value, "value", value)
        return resolved if isinstance(resolved, str) else None

    def _rewrite_w4a8_chunked_finalize_rs(self, graph: fx.Graph) -> int:
        if not self.w4a8_chunked_layers:
            return 0

        tp_group = get_tp_group()
        matched = 0
        for reduce_scatter in list(graph.nodes):
            if (
                reduce_scatter.op != "call_function"
                or reduce_scatter.target != torch.ops.vllm.reduce_scatter.default
                or len(reduce_scatter.args) < 4
            ):
                continue
            add, dim, world_size, group_name = reduce_scatter.args[:4]
            if (
                not isinstance(add, fx.Node)
                or dim != 0
                or world_size != 4
                or group_name != tp_group.unique_name
                or len(add.users) != 1
                or add.op != "call_function"
                or add.target != torch.ops.aten.add.Tensor
                or add.kwargs.get("alpha", 1) != 1
                or len(add.args) < 2
            ):
                continue

            lhs, rhs = add.args[:2]
            if not isinstance(lhs, fx.Node) or not isinstance(rhs, fx.Node):
                continue
            if (
                lhs.op != "call_function"
                or rhs.op != "call_function"
                or lhs.target is not operator.getitem
                or rhs.target is not operator.getitem
                or len(lhs.args) != 2
                or len(rhs.args) != 2
                or {lhs.args[1], rhs.args[1]} != {0, 1}
                or lhs.args[0] is not rhs.args[0]
                or len(lhs.users) != 1
                or len(rhs.users) != 1
            ):
                continue

            moe = lhs.args[0]
            if (
                not isinstance(moe, fx.Node)
                or moe.op != "call_function"
                or moe.target != torch.ops.vllm.moe_forward_shared.default
                or set(moe.users) != {lhs, rhs}
                or len(moe.args) != 6
                or moe.args[0] is not moe.args[2]
                or moe.args[3] is not None
                or moe.args[5] != 0
            ):
                continue

            layer_name = self._resolve_hoisted_layer_name(moe.args[4])
            routed_scale = (
                self.w4a8_chunked_layers.get(layer_name)
                if layer_name is not None
                else self.w4a8_chunked_wildcard_scale
            )
            if routed_scale is None:
                continue

            output_meta = reduce_scatter.meta.get("val")
            if (
                output_meta is None
                or getattr(output_meta, "dtype", None) != torch.bfloat16
                or getattr(output_meta, "ndim", None) != 2
                or output_meta.shape[1] != 6144
            ):
                continue

            with graph.inserting_before(reduce_scatter):
                replacement = graph.call_function(
                    torch.ops.vllm.moe_forward_shared_w4a8_chunked_finalize_rs.default,
                    args=(
                        *moe.args,
                        routed_scale,
                        256,
                        4,
                        "compiler_sp_async_tp_w4a8",
                    ),
                )
                replacement.meta = dict(reduce_scatter.meta)

            reduce_scatter.replace_all_uses_with(replacement)
            graph.erase_node(reduce_scatter)
            graph.erase_node(add)
            graph.erase_node(lhs)
            graph.erase_node(rhs)
            graph.erase_node(moe)
            matched += 1

        if matched:
            graph.lint()
        return matched

    def is_applicable_for_range(self, compile_range: Range) -> bool:
        # This pass is applied on top of the sequence parallelism pass,
        # which is only supported in fullgraph compilation mode.
        assert (
            self.compilation_config.use_inductor_graph_partition
            or not self.compilation_config.splitting_ops
        ), "AsyncTPPass requires full-graph compilation"
        return True

    @VllmInductorPass.time_and_log
    def __call__(self, graph: fx.Graph) -> None:
        w4a8_matches = self._rewrite_w4a8_chunked_finalize_rs(graph)
        self.matched_count = w4a8_matches + self.pm_pass.apply(graph)
        VllmPatternMatcherPass.match_table[self.pass_name] += self.matched_count
        logger.debug("Replaced %s W4A8 MoE finalize-RS patterns", w4a8_matches)
        logger.debug("Replaced %s patterns", self.matched_count)
