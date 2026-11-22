# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from vllm import envs
from vllm.engine.protocol import EngineClient
from vllm.entrypoints.serve.tokenize.serving import ServingTokenization
from vllm.logger import init_logger
from vllm.version import __version__ as VLLM_VERSION

router = APIRouter()

logger = init_logger(__name__)


def base(request: Request) -> ServingTokenization:
    # Reuse the existing instance
    return tokenization(request)


def tokenization(request: Request) -> ServingTokenization:
    return request.app.state.serving_tokenization


def engine_client(request: Request) -> EngineClient:
    return request.app.state.engine_client


@router.get("/load")
async def get_server_load_metrics(request: Request):
    # This endpoint returns the current server load metrics.
    # It tracks requests utilizing the GPU from the following routes:
    # - /v1/responses
    # - /v1/responses/{response_id}
    # - /v1/responses/{response_id}/cancel
    # - /v1/messages
    # - /v1/chat/completions
    # - /v1/completions
    # - /v1/audio/transcriptions
    # - /v1/audio/translations
    # - /v1/embeddings
    # - /pooling
    # - /classify
    # - /score
    # - /v1/score
    # - /rerank
    # - /v1/rerank
    # - /v2/rerank
    return JSONResponse(content={"server_load": request.app.state.server_load_metrics})


@router.get("/version")
async def show_version():
    ver = {"version": VLLM_VERSION}
    return JSONResponse(content=ver)


@router.post("/internal/prefill/progress")
async def get_prefill_progress(request: Request):
    if not envs.VLLM_ENABLE_PREFILL_PROGRESS:
        raise HTTPException(status_code=404, detail="Prefill progress is disabled")
    body = await request.json()
    ids = body.get("request_ids") if isinstance(body, dict) else None
    if (
        not isinstance(ids, list)
        or len(ids) > 128
        or any(not isinstance(value, str) or len(value) > 256 for value in ids)
    ):
        raise HTTPException(status_code=400, detail="Expected up to 128 request IDs")
    return {"engines": await engine_client(request).get_prefill_progress(ids)}


@router.get("/kv_event_sources")
async def kv_event_sources(raw_request: Request):
    """KV-event publisher config per DP rank; empty on render-only servers."""
    client: EngineClient | None = raw_request.app.state.engine_client
    return client.get_kv_event_sources() if client is not None else {}


@router.post("/internal/decode/capacity")
async def get_decode_capacity(request: Request):
    if not envs.VLLM_ENABLE_PREFILL_PROGRESS:
        raise HTTPException(status_code=404, detail="Request progress is disabled")
    body = await request.json()
    requested = body.get("request_ids_by_rank") if isinstance(body, dict) else None
    if not isinstance(requested, dict) or len(requested) > 64:
        raise HTTPException(status_code=400, detail="Expected request IDs by rank")
    client = engine_client(request)
    if client is None:
        raise HTTPException(status_code=503, detail="Engine is unavailable")
    rank_count = client.vllm_config.parallel_config.data_parallel_size

    def parse_groups(value):
        if not isinstance(value, dict) or len(value) > 64:
            raise HTTPException(status_code=400, detail="Invalid capacity query")
        grouped = {}
        for rank, ids in value.items():
            try:
                numeric_rank = int(rank)
            except (TypeError, ValueError) as exc:
                raise HTTPException(status_code=400, detail="Invalid rank") from exc
            if (
                numeric_rank < 0
                or numeric_rank >= rank_count
                or str(numeric_rank) != rank
                or not isinstance(ids, list)
                or len(ids) > 128
                or any(not isinstance(item, str) or len(item) > 256 for item in ids)
            ):
                raise HTTPException(status_code=400, detail="Invalid capacity query")
            grouped[numeric_rank] = ids
        return grouped

    grouped = parse_groups(requested)
    closing = parse_groups(body.get("close_request_ids_by_rank", {}))
    epochs = body.get("expected_epochs_by_rank", {})
    if not isinstance(epochs, dict) or any(
        str(rank) not in epochs
        or not isinstance(epochs[str(rank)], str)
        or len(epochs[str(rank)]) != 32
        for rank in closing
    ):
        raise HTTPException(status_code=400, detail="Closing requires each rank epoch")
    expected = {rank: epochs[str(rank)] for rank in closing}
    return {"engines": await client.get_decode_capacity(grouped, closing, expected)}
