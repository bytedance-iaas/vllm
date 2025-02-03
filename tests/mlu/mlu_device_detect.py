import torch
import torch_mlu

print("PyTorch version:", torch.__version__)  # 2.5.0
print("CUDA version:", torch.version.cuda)  # Output CUDA version 
print("CUDA available:", torch.cuda.is_available())  # Check device/driver/lib: True
# print("GPU:", torch.cuda.get_device_name(0))  # Output GPU name

########################
# Require torch_mlu
# Check mlu dev
########################
print("torch_mlu version:", torch_mlu.__version__)  # 1.24.1-torch2.5.0
print("torch device type = ", torch.device("mlu").type)
if torch.device("mlu").type == "mlu":
    print("MLU device is available")
    print("Device name:", torch.mlu.get_device_name(0))  # Get MLU dev name
else:
    print("MLU device is not available")

if torch.mlu.is_available():
    device_count = torch.mlu.device_count() # According to on MLU_VISIBLE_DEVICES
    print("MLU Device Count: ", device_count)
    print("MLU Devices:")
    for i in range(device_count):
        print(f"MLU Device {i}: {torch.mlu.get_device_name(i)}")
    #driver_version = torch.mlu.get_driver_version()
    #print(f"MLU Driver Version: {driver_version}")
    #lib_version = torch.mlu.get_lib_version()
    #print(f"MLU Library Version: {lib_version}")
