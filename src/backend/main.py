import asyncio
import json
import logging
from collections.abc import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from .agent import NebiusAgent
from .config import get_settings
from .logging_config import configure_logging
from .replay import SyntheticReplay
from .schemas import AgentPingResponse, AgentStatus, ReplayFile, SafetyGateState, TelemetryFrame

configure_logging()
logger = logging.getLogger(__name__)
settings = get_settings()
replay_builder = SyntheticReplay(settings)
agent = NebiusAgent(settings)
app = FastAPI(title="Swatterfly Phase 1 Backend", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origin_list, allow_credentials=False, allow_methods=["GET"], allow_headers=["*"])


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "mode": "synthetic-read-only"}


@app.get("/telemetry/snapshot", response_model=ReplayFile)
def telemetry_snapshot() -> ReplayFile:
    return replay_builder.build()


@app.get("/safety-gate", response_model=SafetyGateState)
def safety_gate() -> SafetyGateState:
    return SafetyGateState()


@app.get("/telemetry/stream")
async def telemetry_stream(request: Request) -> StreamingResponse:
    replay = replay_builder.build()

    async def events() -> AsyncIterator[str]:
        for frame in replay.frames:
            if await request.is_disconnected():
                break
            yield f"data: {frame.model_dump_json()}\\n\\n"
            await asyncio.sleep(settings.replay_interval_seconds)

    return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "Connection": "keep-alive"})


@app.get("/agent/status", response_model=AgentStatus)
def agent_status() -> AgentStatus:
    return agent.status()


@app.get("/agent/ping", response_model=AgentPingResponse)
async def agent_ping() -> AgentPingResponse:
    return await agent.ping()
