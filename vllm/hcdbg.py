"""
Usage: import vllm.hcdbg as hcdbg
"""
# 
import inspect
import traceback

# 控制 jack_print 是否啟用
JACK_PRINT_ENABLED = 1

def jack_print(*args, **kwargs):
    """單純控制開關的 print，不加 prefix"""
    if JACK_PRINT_ENABLED:
        print(*args, **kwargs)

def hcdbg_print(*args, **kwargs):
    """固定前綴 'hcdbg:' 的 print"""
    print("hcdbg:", *args, **kwargs)

def prefix_print(*args, **kwargs):
    """帶有檔案、行號、函數名稱的 debug print"""
    frame = sys._getframe(1)  # 取得上一層函數的 frame
    filename = frame.f_code.co_filename.split("/")[-1]  # 取得檔案名稱
    lineno = frame.f_lineno  # 取得行號
    func_name = frame.f_code.co_name  # 取得函數名稱

    # 組合 debug prefix
    prefix = f"hcdbg[{filename}:{lineno}] {func_name}() -"
    print(prefix, *args, **kwargs)


def dump_stack(limit=5):
    stack = traceback.extract_stack()[:-1]  # 移除 dump_stack 自己
    stack = stack[-limit:]  # 只保留最後 limit 層
    for filename, lineno, function, _ in stack:
        print(f"Function: {function} (File: {filename}, Line: {lineno})")

        
def debug_print(msg):
    frame = inspect.currentframe().f_back  # 獲取上一層調用者的frame
    print(f"hcdbg[{frame.f_code.co_filename}:{frame.f_lineno}] {frame.f_code.co_name}() - {msg}")



# 嘗試寫一個general看device是哪種的function甚至return device name
import torch

def get_device_info():
    """
    這邊再啟動時候有看device為何
    INFO 02-19 13:11:21 __init__.py:190] Automatically detected platform cuda.
    (vllm/platforms/__init__.py)
    builtin_platform_plugins = {
    'tpu': tpu_platform_plugin,
    'cuda': cuda_platform_plugin,
    'rocm': rocm_platform_plugin,
    'hpu': hpu_platform_plugin,
    'xpu': xpu_platform_plugin,
    'cpu': cpu_platform_plugin,
    'neuron': neuron_platform_plugin,
    'openvino': openvino_platform_plugin,
    }
    """
    if torch.cuda.is_available():
        device_type = "CUDA (NVIDIA GPU)"
        device_name = torch.cuda.get_device_name(0)
        device_count = torch.cuda.device_count()
    elif hasattr(torch, "backends") and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device_type = "MPS (Apple GPU)"
        device_name = "Apple Metal"
        device_count = 1  # Apple MPS 目前只支援單 GPU
    elif torch.has_mlu:
        device_type = "MLU (Cambricon)"
        device_name = "寒武紀 MLU"
        device_count = 1  # 具體可用數量需要用寒武紀的 API 確認
    elif torch.has_xpu:
        device_type = "XPU (Intel GPU)"
        device_name = "Intel XPU"
        device_count = 1  # 具體數量也可以額外確認
    elif torch.has_ipu:
        device_type = "IPU (Graphcore)"
        device_name = "Graphcore IPU"
        device_count = 1
    elif torch.has_hpu:
        device_type = "HPU (Habana Gaudi)"
        device_name = "Habana Gaudi"
        device_count = 1
    else:
        device_type = "CPU"
        device_name = "Generic CPU"
        device_count = 1

    print(f"Device Type: {device_type}")
    print(f"Device Name: {device_name}")
    print(f"Device Count: {device_count}")


# 範例
# from vllm.model_executor import utils
# print(f'jack vllm/model_executor/utils has = {dir(utils)}')
# import sys
# print(f'qqq {sys.path}')