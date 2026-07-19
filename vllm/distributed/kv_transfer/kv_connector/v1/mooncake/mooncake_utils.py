# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import threading
import time
from dataclasses import dataclass, field

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from vllm.config import ParallelConfig
from vllm.distributed.kv_transfer.kv_connector.utils import EngineId
from vllm.logger import init_logger

WorkerAddr = str

logger = init_logger(__name__)


def get_mooncake_dp_engine_index(parallel_config: ParallelConfig) -> int:
    """Return the per-engine DP index used for Mooncake side channels."""
    if parallel_config.local_engines_only:
        assert parallel_config.data_parallel_rank_local is not None
        return parallel_config.data_parallel_rank_local

    return parallel_config.data_parallel_index


class RegisterWorkerPayload(BaseModel):
    engine_id: EngineId
    dp_rank: int
    tp_rank: int
    tp_size: int = 1
    pp_rank: int
    pp_size: int = 1
    pcp_rank: int = 0
    pcp_size: int = 1
    dcp_size: int = 1
    addr: WorkerAddr


@dataclass
class EngineEntry:
    engine_id: EngineId
    # Canonical PCP=0 workers, preserving the pre-PCP response shape.
    # {tp_rank: {pp_rank: worker_addr}}
    worker_addr: dict[int, dict[int, WorkerAddr]]
    pcp_size: int
    dcp_size: int
    tp_size: int = 1
    pp_size: int = 1
    # {tp_rank: {pp_rank: {pcp_rank: worker_addr}}}
    pcp_worker_addr: dict[int, dict[int, dict[int, WorkerAddr]]] = field(
        default_factory=dict
    )

    def is_complete(self) -> bool:
        expected_tp_ranks = set(range(self.tp_size))
        if set(self.pcp_worker_addr) != expected_tp_ranks:
            return False

        expected_pp_ranks = set(range(self.pp_size))
        expected_pcp_ranks = set(range(self.pcp_size))
        return all(
            set(tp_entry) == expected_pp_ranks
            and all(
                set(pp_entry) == expected_pcp_ranks for pp_entry in tp_entry.values()
            )
            for tp_entry in self.pcp_worker_addr.values()
        )


class MooncakeBootstrapServer:
    """
    A centralized server running on the global rank 0 prefiller worker.
    Prefiller workers register their connection info (IP, port, ranks) here.
    """

    def __init__(self, host: str, port: int):
        self.workers: dict[int, EngineEntry] = {}

        self.host = host
        self.port = port
        self.app = FastAPI()
        self._register_routes()
        self.server_thread: threading.Thread | None = None
        self.server: uvicorn.Server | None = None

    def __del__(self):
        self.shutdown()

    def _register_routes(self):
        # All methods are async. No need to use lock to protect data.
        self.app.post("/register")(self.register_worker)
        self.app.get("/query", response_model=dict[int, EngineEntry])(self.query)

    def start(self):
        if self.server_thread:
            return

        config = uvicorn.Config(app=self.app, host=self.host, port=self.port)
        self.server = uvicorn.Server(config=config)
        self.server_thread = threading.Thread(
            target=self.server.run, name="mooncake_bootstrap_server", daemon=True
        )
        self.server_thread.start()
        while not self.server.started:
            time.sleep(0.1)  # Wait for the server to start
        logger.info("Mooncake Bootstrap Server started at %s:%d", self.host, self.port)

    def shutdown(self):
        if self.server_thread is None or self.server is None or not self.server.started:
            return

        self.server.should_exit = True
        self.server_thread.join()
        logger.info("Mooncake Bootstrap Server stopped.")

    async def register_worker(self, payload: RegisterWorkerPayload):
        """Handles registration of a prefiller worker."""
        if (
            payload.tp_size < 1
            or payload.pp_size < 1
            or payload.pcp_size < 1
            or payload.dcp_size < 1
        ):
            raise HTTPException(
                status_code=400,
                detail="TP, PP, PCP, and DCP sizes must be positive.",
            )
        if payload.tp_rank < 0 or payload.tp_rank >= payload.tp_size:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"TP rank {payload.tp_rank} must be less than TP size "
                    f"{payload.tp_size}."
                ),
            )
        if payload.pp_rank < 0 or payload.pp_rank >= payload.pp_size:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"PP rank {payload.pp_rank} must be less than PP size "
                    f"{payload.pp_size}."
                ),
            )
        if payload.pcp_rank < 0 or payload.pcp_rank >= payload.pcp_size:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"PCP rank {payload.pcp_rank} must be less than PCP size "
                    f"{payload.pcp_size}."
                ),
            )
        if payload.pcp_size > 1 and payload.dcp_size != 1:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Mooncake canonical PCP replica transfer requires DCP size 1, "
                    f"but got PCP size {payload.pcp_size} and DCP size "
                    f"{payload.dcp_size}"
                ),
            )
        if payload.dp_rank not in self.workers:
            self.workers[payload.dp_rank] = EngineEntry(
                engine_id=payload.engine_id,
                tp_size=payload.tp_size,
                pp_size=payload.pp_size,
                pcp_size=payload.pcp_size,
                dcp_size=payload.dcp_size,
                worker_addr={},
            )

        dp_entry = self.workers[payload.dp_rank]
        if dp_entry.engine_id != payload.engine_id:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Engine ID mismatch for dp_rank={payload.dp_rank}: "
                    f"expected {dp_entry.engine_id}, got {payload.engine_id}"
                ),
            )
        if dp_entry.tp_size != payload.tp_size:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"TP size mismatch for dp_rank={payload.dp_rank}: "
                    f"expected {dp_entry.tp_size}, got {payload.tp_size}"
                ),
            )
        if dp_entry.pp_size != payload.pp_size:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"PP size mismatch for dp_rank={payload.dp_rank}: "
                    f"expected {dp_entry.pp_size}, got {payload.pp_size}"
                ),
            )
        if dp_entry.pcp_size != payload.pcp_size:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"PCP size mismatch for dp_rank={payload.dp_rank}: "
                    f"expected {dp_entry.pcp_size}, got {payload.pcp_size}"
                ),
            )
        if dp_entry.dcp_size != payload.dcp_size:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"DCP size mismatch for dp_rank={payload.dp_rank}: "
                    f"expected {dp_entry.dcp_size}, got {payload.dcp_size}"
                ),
            )
        if payload.tp_rank not in dp_entry.pcp_worker_addr:
            dp_entry.pcp_worker_addr[payload.tp_rank] = {}

        tp_entry = dp_entry.pcp_worker_addr[payload.tp_rank]
        if payload.pp_rank not in tp_entry:
            tp_entry[payload.pp_rank] = {}

        pp_entry = tp_entry[payload.pp_rank]
        if payload.pcp_rank in pp_entry:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Worker with dp_rank={payload.dp_rank}, "
                    f"tp_rank={payload.tp_rank}, pp_rank={payload.pp_rank}, "
                    f"pcp_rank={payload.pcp_rank} "
                    f"is already registered at "
                    f"{pp_entry[payload.pcp_rank]}, "
                    f"but still want to register at {payload.addr}"
                ),
            )

        pp_entry[payload.pcp_rank] = payload.addr
        if payload.pcp_rank == 0:
            dp_entry.worker_addr.setdefault(payload.tp_rank, {})[payload.pp_rank] = (
                payload.addr
            )
        logger.debug(
            "Registered worker: engine_id=%s, dp_rank=%d, tp_rank=%d, "
            "pp_rank=%d, pcp_rank=%d at %s",
            payload.engine_id,
            payload.dp_rank,
            payload.tp_rank,
            payload.pp_rank,
            payload.pcp_rank,
            payload.addr,
        )

        return {"status": "ok"}

    async def query(self) -> dict[int, EngineEntry]:
        if not self.workers or any(
            not entry.is_complete() for entry in self.workers.values()
        ):
            raise HTTPException(
                status_code=503,
                detail="Mooncake prefiller worker topology is not ready.",
            )
        return self.workers
