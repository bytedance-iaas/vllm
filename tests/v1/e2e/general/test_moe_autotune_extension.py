# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from pathlib import Path

import httpx
import pytest

from tests.utils import RemoteOpenAIServer

MODEL = os.environ.get("VLLM_TEST_MOE_CACHE_MODEL", "facebook/opt-125m")
EXTENSION = "vllm.plugins.tk300_moe_cache.MoEAutotuneCacheWorker"


@pytest.mark.parametrize("configured", [False, True])
def test_unqualified_moe_cache_preserves_real_generation(tmp_path: Path, configured):
    """Disabled or unsupported replay leaves the public serving path unchanged."""
    args = ["--max-model-len", "256", "--enforce-eager"]
    body = dict(
        model=MODEL,
        prompt="The capital of France is",
        max_tokens=8,
        temperature=0,
        seed=0,
    )
    with RemoteOpenAIServer(MODEL, args) as server:
        expected = server.get_client().completions.create(**body).choices[0].text

    env = {
        "VLLM_SERVER_DEV_MODE": "1",
        "TK300_MOE_AUTOTUNE_MANIFEST": str(tmp_path / "missing.json")
        if configured
        else "",
    }
    with RemoteOpenAIServer(
        MODEL, args + ["--worker-extension-cls", EXTENSION], env_dict=env
    ) as server:
        actual = server.get_client().completions.create(**body).choices[0].text
        response = httpx.post(
            server.url_for("collective_rpc"),
            json={"method": "moe_autotune_cache_status"},
            timeout=30,
        )
        response.raise_for_status()
        states = response.json()["results"]
        assert states and all(not state["loaded"] for state in states)
        assert actual == expected and actual.strip()
