from abc import ABC, abstractmethod
from vllm.envs import VLLM_GPU_RDMA_MAP
from urllib.parse import urlparse
from hpkv_prefix import client

def parse_hpkv_url(url):
    parsed = urlparse(url)
    if parsed.scheme != 'hpkv':
        raise ValueError("Invalid URL scheme. Expected 'hpkv'.")
    
    netloc = parsed.netloc
    if ':' in netloc:
        ip, port = netloc.split(':', 1)
        try:
            port = int(port)
        except ValueError:
            raise ValueError("Invalid port number. Port must be a valid integer.")
    else:
        ip = netloc
        port = None
    
    return ip, port

def get_swapper_class(uri: str, local_rank: int = 0):
    # hpkv://ip:port
    if uri.startswith("hpkv"):
        ip, port = parse_hpkv_url(uri)
        local_rdma_ip = VLLM_GPU_RDMA_MAP[local_rank]
        return HPKVSwapper(ip, port, local_rdma_ip, 0, local_rank)
    else:
        raise RuntimeError("only support hpkv and cpu swapper.")

class SwapperBase(ABC):
    @abstractmethod
    def swap_in_mha(self, block_mapping) -> None:
        raise NotImplementedError
    
    @abstractmethod
    def swap_out_mha(self, block_mapping) -> None:
        raise NotImplementedError
    
    @abstractmethod
    def exist(self, key) -> bool:
        raise NotImplementedError
    
    @abstractmethod
    def reg_mr(self, tensors):
        raise NotImplementedError
    
    @abstractmethod
    def get_loaded_reqs(self):
        raise NotImplementedError

    @abstractmethod
    def get_saved_blocks(self):
        raise NotImplementedError


class HPKVSwapper(SwapperBase):
    def __init__(self, rip, rport, lip, lport, rank = 0):
        self.swapper = client.kvCacheManager(rip, rport, lip, lport, rank)

    def swap_in_mha(self, req_id, block_mapping) -> None:
        self.swapper.swap_in(req_id, block_mapping)

    def swap_out_mha(self, block_mapping) -> None:
        self.swapper.swap_out(block_mapping)

    def exist(self, key) -> bool:
        return self.swapper.test(key)
    
    def reg_mr(self, tensors):
        self.swapper.reg_mr(tensors)

    def get_loaded_reqs(self):
        return self.swapper.get_loaded_reqs()

    def get_saved_blocks(self):
        return self.swapper.get_saved_blocks()
