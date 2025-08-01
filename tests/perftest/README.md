# perftest
The performance test suite for vllm

Overview
Modified from vllm/benchmarks/benchmark-serving. Test case based perf test suite for vllm.

Test Environment
Required components
1. Infinistore
2. vllm prefill server
3. vllm decode server
4. vllm kv proxy

Run Perftest
In order to run the perftest.py, you must either specify test case json file 
from command line like the following:
        --test-cases path-to-your-test-case-json-file

or you must have the test case json file named as "test_cases.json" in
./benchmarks directdory.



A typical test case looks like:
```
{
  "tests": [
    {
      "test-id": 1,
      "enabled": true,
      "group": "PD分离",
      "name": "H20-PD",
      "description": "",
      "program": "benchmark",
      "backend": "openai-chat",
      "model": "/infinistore/Llama-3.1-8B-Instruct",
      "base-url": "http://127.0.0.1:8888",
      "endpoint": "/v1/chat/completions",
      "num-prompts": 10,
      "request-rate": 2,
      "percentile-metrics": "ttft,tpot,itl",
      "metric-percentiles": "50,90,95,99",
      "goodput": {
        "ttft": 5000,
        "tpot": 50
      },
      "max-concurrency": 200,
      "random-input-len": 2048,
      "random-output-len": 200,
      "random-prefix-len": 0,
      "random-range-ratio": 1.0,
      "dataset-name": "random",
      "dataset-path": "",
      "ignore-eos" : true,
      "gpu-metric-interval": 0.5,
      "profile": false,
      "tokenizer": ""
    },
    {
      "test-id": 2,
      "enabled": false,
      "group": "PD分离",
      "name": "4090+H20-PD",
      "description": "",
      "program": "benchmark",
      "backend": "vllm",
      "model": "/data00/models/Meta-Llama-3.1-8B-Instruct",
      "base-url": "http://127.0.0.1:8099",
      "endpoint": "/v1/completions",
      "num-prompts": 200,
      "request-rate": 4.5,
      "percentile-metrics": "ttft,tpot,itl",
      "metric-percentiles": "50,90,95,99",
      "goodput": {
        "ttft": 5000,
        "tpot": 50
      },
      "max-concurrency": 200,
      "sharegpt-output-len": 200,
      "dataset-name": "sharegpt",
      "ignore-eos" : true,
      "gpu-metric-interval": 0.5,
      "tokenizer": ""
    },
    {
      "test-id": 3,
      "enabled": false,
      "group": "PD分离",
      "name": "4090+H20-PD-FP8",
      "program": "benchmark",
      "description": "",
      "backend": "tensorrt-llm",
      "model": "ensemble",
      "base-url": "http://localhost:8099",
      "endpoint": "/v2/models/ensemble/generate_stream",
      "num-prompts": 200,
      "request-rate": 10,
      "percentile-metrics": "ttft,tpot,itl",
      "metric-percentiles": "50,90,95,99",
      "goodput": {
        "ttft": 5000,
        "tpot": 50
      },
      "max-concurrency": 200,
      "sonnet-input-len": 2048,
      "sonnet-output-len": 1,
      "sonnet-prefix-len": 0,
      "dataset-name": "sonnet",
      "ignore-eos" : true,
      "tokenizer": "/data00/models/Meta-Llama-3.1-70B-Instruct/",
      "gpu-metric-interval": 0.5,
      "trust-remote-code": true
    },
    {
      "test-id": 4,
      "enabled": false,
      "group": "PrefixCache",
      "name": "混合部署测试",
      "description": "固定所有promopt中存在25%相同内容",
      "program": "benchmark",
      "backend": "vllm",
      "model": "/data00/models/Meta-Llama-3.1-8B-Instruct",
      "base-url": "http://localhost:8099",
      "endpoint": "/v1/completions",
      "num-prompts": 200,
      "request-rate": 1,
      "percentile-metrics": "ttft,tpot,itl",
      "metric-percentiles": "50,90,95,99",
      "goodput": {
        "ttft": 5000,
        "tpot": 50
      },
      "max-concurrency": 200,
      "hf-subset": "",
      "hf-split": "",
      "hf-output-len": 200,
      "dataset-name": "hf",
      "ignore-eos" : true,
      "random-prefix-len": 512,
      "gpu-metric-interval": 0.5,
      "tokenizer": ""
    }
  ]
}
```

Datasets supported
1. random (tested)

2. sharegpt (not tested)

3. sonnet (not tested)

4. hf (not tested)

