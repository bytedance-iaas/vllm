import torch
from torch import nn

class SupportCpuOffLoadWeight:
    def __init__(self, sub_module_list, var_index):
        self.sub_module_list = sub_module_list
        self.var_index = var_index

    def __call__(self, forward_method):
        def forward_wrapper(module, *args, **kwargs):
            moved_weights = []
            for module_name in self.sub_module_list:
                module_instance = getattr(module, module_name)
                if not isinstance(module_instance, nn.Module):
                    raise TypeError(f"{module_name} is not an instance of nn.Module")
                
                if isinstance(self.var_index, int) :
                    if len(args) <= self.var_index:
                        raise AttributeError(f"forward function does not have parameters #{self.var_index+1}, args: {args}, kwarags: {kwargs.keys()}")
                    else:
                        target_tensor = args[self.var_index]
                elif isinstance(self.var_index, str):
                    if self.var_index not in kwargs.keys():
                        raise AttributeError(f"forward function does not have parameter named {self.var_index}, args: {args}, kwarags: {kwargs.keys()}")
                    else:
                        target_tensor = kwargs[self.var_index]
                else:
                    raise AttributeError(f"var_index is not an integer or string")

                
                if isinstance(target_tensor, torch.Tensor) == False:
                    raise TypeError(f"target_tensor is not an instance of torch.Tensor")
                
                for module_name in self.sub_module_list:
                    module_instance = getattr(module, module_name)
                    # if module_instance.weight.device.type == 'cpu':
                    print(f"Moving weight of {module_name} to device")
                    moved_weights.append(module_instance)
                    module_instance.to(device=target_tensor.device)

            result = forward_method(module, *args, **kwargs)

            for weight in moved_weights:
                weight.to('cpu')
            torch.cuda.empty_cache()
            return result
        
        return forward_wrapper