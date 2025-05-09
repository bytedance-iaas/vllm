from vllm.config import SchedulerConfig
from vllm.engine.arg_utils import EngineArgs
from vllm.utils import FlexibleArgumentParser


def test_engine_args_direct():
    # Default should be False
    args = EngineArgs()
    assert not args.enable_prefill_optimizer

    # Explicit True
    args = EngineArgs(enable_prefill_optimizer=True)
    assert args.enable_prefill_optimizer


def test_engine_args_cli_flag():
    parser = FlexibleArgumentParser()
    parser = EngineArgs.add_cli_args(parser)

    # Flag present → should be True
    parsed = parser.parse_args(["--enable-prefill-optimizer"])
    args = EngineArgs.from_cli_args(parsed)
    assert args.enable_prefill_optimizer

    # Flag absent → should be False
    parsed = parser.parse_args([])
    args = EngineArgs.from_cli_args(parsed)
    assert not args.enable_prefill_optimizer


def test_scheduler_config_propagation():
    args = EngineArgs(enable_prefill_optimizer=True)
    config = SchedulerConfig(enable_prefill_optimizer=args.enable_prefill_optimizer)
    assert config.enable_prefill_optimizer

    args = EngineArgs()
    config = SchedulerConfig(enable_prefill_optimizer=args.enable_prefill_optimizer)
    assert not config.enable_prefill_optimizer
