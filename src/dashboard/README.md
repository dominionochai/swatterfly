# Swatterfly Phase 1 cockpit

This is the minimal Next.js/TypeScript implementation of `docs/06-cockpit.md`. It is intentionally read-only: the browser opens an SSE stream or a deterministic snapshot and has no command, actuator, or flight-authority API path.

## Run

```bash
cd src/dashboard
npm install
NEXT_PUBLIC_BACKEND_URL=http://localhost:8000 npm run dev
```

Open http://localhost:3000 after starting the backend. `DEMO REPLAY` fetches `/telemetry/snapshot` and plays its timestamped frames in slow motion; `RETURN LIVE` reconnects to the SSE stream. If npm installation/network is unavailable, the source and configuration remain committed but build verification is blocked.

No external UI library is used. The event-cam is a canvas placeholder with moving synthetic dots, lock box, and explanatory overlay markers.
