from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class TelemetryFrame(BaseModel):
    """Synthetic, read-only telemetry contract; values are not sensor measurements."""

    frame_id: str
    replay_id: str
    theta: float = Field(description="Synthetic angular-size signal, degrees")
    eta: float = Field(description="Synthetic looming activity, normalized 0..1")
    tau: float = Field(description="Synthetic time-to-contact countdown, seconds")
    target_lock_state: Literal["SEARCHING", "LOCKED", "COASTING"]
    confidence: float = Field(ge=0, le=1)
    timestamp: datetime
    closing_speed_mps: float
    battery_percent: float = Field(ge=0, le=100)
    gps: dict[str, float]
    threat_log: list[str]
    line_of_sight_deg: float
    bearing_rate_dps: float
    predicted_intercept_point: dict[str, float]
    acceleration_vector: dict[str, float]
    lgmd_spike: float = Field(ge=0, le=1)


class ReplayFile(BaseModel):
    replay_id: str
    seed: int
    mode: Literal["synthetic-demo"] = "synthetic-demo"
    generated_at: datetime
    frames: list[TelemetryFrame]


class SafetyGateState(BaseModel):
    """Placeholder only: Phase 1 exposes no flight authority or actuator path."""

    state: Literal["READ_ONLY_PLACEHOLDER"] = "READ_ONLY_PLACEHOLDER"
    command_path_enabled: bool = False
    reason: str = "No flight commands are implemented in Phase 1."


class AgentStatus(BaseModel):
    provider: str = "Nebius Token Factory"
    base_url: str
    configured: bool
    primary_model: str
    fallback_model: str
    message: str


class AgentPingResponse(BaseModel):
    status: str
    model: str | None = None
    response: str | None = None
    message: str
