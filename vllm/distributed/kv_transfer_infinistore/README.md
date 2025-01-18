# Infinistore Based Prefill-Decode Disaggration Inference

This folder implements distributed KV cache transfer based on Infinistore. (https://github.com/bd-iaas-us/infiniStore)

Note that this implementation requires the hosts be equipped with RDMA.

## Steps to Start 

The prefill & decode inference engines can be in one host or across a network. If running both engines in one host, make sure each engine use difference GPUs via proper configuration of environment variable CUDA_VISIBLE_DEVICES.

1. Installation

In the prefill side, install infinistore:
```
pip install infinistore
```

In both sides, install vllm from branch https://github.com/bd-iaas-us/vllm/tree/pd-disaggration-main.

```
cd [repo_root]
git checkout pd-disaggration-main
pip install -e .
```

2. start Infinistore

```
infinistore --log-level warning --dev-name mlx5_0 --link-type Ethernet
```

3. start the inference engines

In the prefill side:

```
PD_SEPARATE_STAGE=prefill INFINISTORE_LOG_LEVEL="warning" INFINISTORE_DEV_NAME="mlx5_0" INFINISTORE_SERVER_ADDR="127.0.0.1:22345" vllm serve [model_path]] --port 8010 --enforce-eager
```

In the decode side:
```
PD_SEPARATE_STAGE=decode INFINISTORE_LOG_LEVEL="info" INFINISTORE_SERVER_ADDR="[prefill_server_IP]:22345" INFINISTORE_DEV_NAME="mlx5_0" INFINISTORE_CONN_TYPE="RDMA" vllm serve [model_path] --uvicorn-log-level debug --port 8020 --enforce-eager
```

4. start the kv transfer proxy

Please manually update DECODE_BASE_URL & PREFILL_BASE_URL in  vllm/distributed/kv_transfer_infinistore/kv_transfer_proxy/kv_proxy.py if the prefill & decode engines are not in the same host or proxy is in a different host. then run:

```
cd [vllm_repo]/vllm/distributed/kv_transfer_infinistore/kv_transfer_proxy
uvicorn kv_proxy:app --host 0.0.0.0 --port 8080 --workers 4
```

5. send request

```
curl http://[proxy_IP]:8080/v1/completions    -H "Content-Type: application/json"   -d '{"model": "[model_path]", "prompt": "San Francisco is a", "max_tokens": 100} '
```
