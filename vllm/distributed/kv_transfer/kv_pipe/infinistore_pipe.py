# SPDX-License-Identifier: Apache-2.0
"""
Simple KV Cache Connector for Distributed Machine Learning Inference

The InfinistoreConnector transfers KV caches between prefill vLLM worker (KV cache 
producer) and decode vLLM worker (KV cache consumer) using PyNcclPipe or
MooncakePipe.

But the logic can be extended to support other pipe and lookup buffer.
"""
from dataclasses import dataclass
import configparser
import os
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

import torch

from vllm import _custom_ops as ops
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.base import KVConnectorBase
from vllm.distributed.kv_transfer.kv_lookup_buffer.simple_buffer import (
    SimpleBuffer)
from vllm.logger import init_logger
from vllm.sequence import IntermediateTensors
from vllm.distributed import (get_tensor_model_parallel_rank,
                              get_tensor_model_parallel_world_size)

if TYPE_CHECKING:
    from vllm.worker.model_runner import ModelInputForGPUWithSamplingMetadata

logger = init_logger(__name__)


class InfinistoreConfig:
    def __init__(
        self,
        host_addr: str,
        service_port: int,
        ib_port: int,
        connect_type: str,
        link_type: str,
        device_name: str,
        log_level: str,
    ):
        self.host_addr = host_addr
        self.service_port = service_port
        self.ib_port = ib_port
        self.connect_type = connect_type
        self.link_type = link_type
        self.device_name = device_name
        self.log_level = log_level

    @staticmethod
    def from_file(infinistore_cfg_file: str, vllm_config: VllmConfig) -> 'InfinistoreConfig':
        """Load the config from a .cfg file and vllm config.
        
        The .cfg file should have the following format:

        [infinitstore]
        host_addr = 127.0.0.1
        service_port = 8100
        connection_type = rdma
        link_type = ethernet
        device_name = mlx5_0
        ib_port = 1
        log_level = error

        """


        config = configparser.ConfigParser()
        config.read(infinistore_cfg_file)

        host_addr = config.get('infinistore', 'host_addr').strip()
        if not host_addr:
            err_msg = "Missing required config value: host_addr"
            logger.error(err_msg)
            raise ValueError(err_msg)
        
        try:
            service_port = config.getint('infinistore', 'service_port')
            if service_port is None:
                raise ValueError
        except ValueError:
            err_msg = "Invalid or missing 'service_port', it must be a valid integer."
            logger.error(err_msg)
            raise ValueError(err_msg)
        
        try:
            ib_port = config.getint('infinistore', 'ib_port')
            if ib_port is None:
                raise ValueError
        except ValueError:
            err_msg = "Invalid or missing 'ib_port', it must be a valid integer."
            logger.error(err_msg)
            raise ValueError(err_msg)
        
        # Read and validate link_type
        link_type = config.get('infinistore', 'link_type', fallback='ethernet').strip().lower()
        if link_type not in InfinistoreConfig.VALID_LINK_TYPES:
            logger.error(f"Invalid 'link_type': {link_type}. Must be one of {InfinistoreConfig.VALID_LINK_TYPES}.")
            raise ValueError(f"Configuration error: 'link_type' must be one of {InfinistoreConfig.VALID_LINK_TYPES}.")

        # Read other fields
        connect_type = config.get('infinistore', 'connect_type', fallback='rdma').strip()
        device_name = config.get('infinistore', 'device_name', fallback='mlx5_0').strip()
        log_level = config.get('infinistore', 'log_level', fallback='error').strip()


        # Log the validated values
        logger.info(f"Loaded InfinistoreConfig: host_addr={host_addr}, service_port={service_port}, "
                    f"connect_type={connect_type}, link_type={link_type}, device_name={device_name}")

        return InfinistoreConfig(
            host_addr=host_addr,
            service_port=service_port,
            ib_port=ib_port,
            connect_type=connect_type,
            link_type=link_type,
            device_name=device_name,
            log_level=log_level,
        )
    
def _is_gdr_supported():
    return True

def _is_rdma_supported():
    return True

    
class InfinistoreTransporter:
    import infinistore
    #Class-level singleton connection instance
    _singleton_rdma_conn = None
    _singleton_local_gpu_conn = None

    def __init__(self,
                 vllm_config: VllmConfig
                 ) -> None:
        
        if not _is_rdma_supported():
            raise ValueError("RDMA is not supported on this system.")

        config_file_path = os.getenv('INFINISTORE_CONFIG_PATH')
        if config_file_path is None:
            raise ValueError(
                "The environment variable 'INFINISTORE_CONFIG_PATH' is not set.")
        infinistore_cfg = InfinistoreConfig.from_file(config_file_path)
        
        if InfinistoreTransporter._singleton_rdma_conn is None:
            infinte_config = infinistore.ClientConfig(
                host_addr=infinistore_cfg.host_addr,
                service_port=infinistore_cfg.service_port,
                log_level=infinistore_cfg.log_level,
                connection_type=infinistore.TYPE_RDMA,
                ib_port=infinistore_cfg.ib_port,
                link_type=infinistore_cfg.link_type,
                dev_name=infinistore_cfg.device_name,
            )
            
            InfinistoreTransporter._singleton_rdma_conn = infinistore.InfinityConnection(
                infinte_config)
            logger.info("Connecting to infinite store server via rdma: %s",
                    infinistore_cfg.host_addr)
            InfinistoreTransporter._singleton_rdma_conn.connect()

            self.rdma_conn = InfinistoreTransporter._singleton_rdma_conn

        if not _is_gdr_supported():
            self.local_gpu_conn = None
        else:
            if InfinistoreTransporter._singleton_local_gpu_conn is None:
                infinte_config = infinistore.ClientConfig(
                host_addr=infinistore_cfg.host_addr,
                service_port=infinistore_cfg.service_port,
                log_level=infinistore_cfg.log_level,
                connection_type=infinistore.TYPE_LOCAL_GPU,
                ib_port=infinistore_cfg.ib_port,
                link_type=infinistore_cfg.link_type,
                dev_name=infinistore_cfg.device_name,
                )
                
                InfinistoreTransporter._singleton_local_gpu_conn = infinistore.InfinityConnection(
                    infinte_config)
                logger.info("Connecting to infinite store server via lcoal gpu: %s",
                        infinistore_cfg.host_addr)
                InfinistoreTransporter._singleton_local_gpu_conn.connect()
            self.local_gpu_conn = InfinistoreTransporter._singleton_local_gpu_conn
        
        self.block_size = vllm_config.cache_config.block_size

        
    def send_tensor(self, source_tensor: torch.Tensor, block_offsets: List[Tuple[str, int]], size: int, rdam_only = False) -> None:

        try:
            if self.local_gpu_conn is not None and not rdam_only:
                self.local_gpu_conn.local_gpu_write_cache(source_tensor, block_offsets,
                                           size)
            else:
                self.conn.rdma_write_cache(source_tensor, block_offsets,
                                            size)
        except Exception as e:
            logger.error("Failed to write kv_cache: %s", e)
            raise e
        
    def recv_tensor(self, target_tensor: torch.Tensor, block_offsets: List[Tuple[str, int]], size: int) -> None:

        try:
            self.rdma_conn.read_cache(target_tensor, block_offsets, size)
        except Exception as e:
            logger.error("Failed to read kv_cache: %s", e)
            raise e
