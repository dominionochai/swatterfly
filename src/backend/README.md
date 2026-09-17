# Swatterfly Phase 1 backend

This is a thin, read-only connected pipe. Every telemetry value is synthetic and deterministic; it does not perform detection, real math, guidance, actuation, or physics tuning. The existing `src/sim`, `src/lgmd`, `src/tracker`, and `src/guidance` placeholders are untouched.

## Run

```bash
cd src/backend
python3.11 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
# from the repository root, or set PYTHONPATH to the repository root:
cd ../..
uvicorn src.backend.main:app --reload --host 127.0.0.1 --port 8000
```

Endpoints:

- `GET /health` — read-only health check
- `GET /telemetry/snapshot` — complete deterministic `ReplayFile`
- `GET /telemetry/stream` — finite SSE synthetic replay
- `GET /safety-gate` — explicit non-authoritative placeholder
- `GET /agent/status` — key configuration status without exposing secrets
- `GET /agent/ping` — trivial GLM-5.3 request through Nebius Token Factory, with GLM-5.2 fallback

Set `NEBIUS_API_KEY` to enable `/agent/ping`. The client uses `NEBIUS_BASE_URL=https://api.tokenfactory.nebius.com/v1/` by default and never reads `OPENAI_API_KEY`. Without a key, the endpoint returns a clear `not_configured` response rather than failing startup. A key-present response must be tested in an environment with an authorized key; this repository does not claim one.

The backend accepts `CORS_ORIGINS` as a comma-separated list and `REPLAY_SEED` for deterministic demos. Copy the root `.env.example` to `.env` for local configuration.
