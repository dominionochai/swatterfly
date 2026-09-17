import asyncio

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from src.backend import main
from src.backend.schemas import TelemetryFrame


client = TestClient(main.app)


def test_telemetry_schema_validates_ranges_and_required_fields() -> None:
    frame = TelemetryFrame(
        frame_id="demo-0001",
        replay_id="demo-20260917",
        theta=2.5,
        eta=0.4,
        tau=3.0,
        target_lock_state="SEARCHING",
        confidence=0.8,
        timestamp="2026-09-17T20:00:00Z",
        closing_speed_mps=3.2,
        battery_percent=95.0,
        gps={"lat": 51.5007, "lon": -0.1246},
        threat_log=["synthetic frame"],
        line_of_sight_deg=-6.0,
        bearing_rate_dps=0.8,
        predicted_intercept_point={"x": 0.38, "y": 0.5},
        acceleration_vector={"x": 0.18, "y": -0.1},
        lgmd_spike=0.3,
    )

    assert frame.target_lock_state == "SEARCHING"
    assert frame.confidence == 0.8

    with pytest.raises(ValidationError):
        TelemetryFrame(
            frame_id="invalid",
            replay_id="demo",
            theta=2.5,
            eta=0.4,
            tau=3.0,
            target_lock_state="SEARCHING",
            confidence=1.1,
            timestamp="2026-09-17T20:00:00Z",
            closing_speed_mps=3.2,
            battery_percent=95.0,
            gps={},
            threat_log=[],
            line_of_sight_deg=0.0,
            bearing_rate_dps=0.0,
            predicted_intercept_point={},
            acceleration_vector={},
            lgmd_spike=0.0,
        )


def test_sse_stream_uses_real_newline_framing() -> None:
    class ConnectedRequest:
        async def is_disconnected(self) -> bool:
            return False

    async def read_frames() -> tuple[str, str]:
        response = await main.telemetry_stream(ConnectedRequest())
        iterator = response.body_iterator
        try:
            connected = await iterator.__anext__()
            data_frame = await iterator.__anext__()
            return connected, data_frame
        finally:
            await iterator.aclose()

    connected, data_frame = asyncio.run(read_frames())

    assert connected == ": connected\nretry: 3000\n\n"
    assert data_frame.startswith("data: ")
    assert data_frame.endswith("\n\n")
    assert "\\n" not in connected
    assert "\\n" not in data_frame


def test_agent_status_and_ping_without_key_are_graceful(monkeypatch) -> None:
    monkeypatch.setattr(main.agent.settings, "nebius_api_key", None)

    status = client.get("/agent/status")
    ping = client.get("/agent/ping")

    assert status.status_code == 200
    assert status.json()["configured"] is False
    assert "NEBIUS_API_KEY" in status.json()["message"]
    assert ping.status_code == 200
    assert ping.json()["status"] == "not_configured"
    assert ping.json()["response"] is None
