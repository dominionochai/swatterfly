# Phase 1 integration runbook

Phase 1 adds a thin connected, read-only pipe using fake telemetry only. Existing `docs/` files and the placeholder directories `src/sim`, `src/lgmd`, `src/tracker`, and `src/guidance` were preserved and not implemented. No actuator, command, flight-authority, physics-tuning, or real detection path is present.

## Run both halves

Terminal 1:

```bash
cp .env.example .env
cd src/backend
python3.11 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
cd ../..
uvicorn src.backend.main:app --reload --host 127.0.0.1 --port 8000
```

Terminal 2:

```bash
cd src/dashboard
npm install
NEXT_PUBLIC_BACKEND_URL=http://localhost:8000 npm run dev
```

Then open `http://localhost:3000`. The dashboard consumes `GET /telemetry/stream` in live mode and `GET /telemetry/snapshot` in slow-motion demo replay mode. The API also exposes `GET /agent/status`, `GET /agent/ping`, and `GET /safety-gate`.

## Nebius GLM behavior

The backend uses `https://api.tokenfactory.nebius.com/v1/`, reads only `NEBIUS_API_KEY`, calls `glm-5.3`, and tries `glm-5.2` if the primary call fails. `/agent/status` reports whether a key is configured without exposing it. With no key, `/agent/ping` returns `not_configured` and a clear setup message. A key-present model response was not tested or claimed in this implementation environment.

## Verification status

Files were committed through the connected GitHub account. No local Python or npm installation/build was run through the GitHub integration, so dependency installation, `uvicorn` startup, and `next build` remain to be verified in a runtime environment. The source is structured for Python 3.11+ and Next.js 14 with minimal dependencies.
