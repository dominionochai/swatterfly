"use client";

import { useCallback, useEffect, useRef, useState } from "react";

type Frame = { frame_id: string; replay_id: string; theta: number; eta: number; tau: number; target_lock_state: "SEARCHING" | "LOCKED" | "COASTING"; confidence: number; timestamp: string; closing_speed_mps: number; battery_percent: number; gps: { lat: number; lon: number }; threat_log: string[]; line_of_sight_deg: number; bearing_rate_dps: number; predicted_intercept_point: { x: number; y: number }; acceleration_vector: { x: number; y: number }; lgmd_spike: number };
type Replay = { replay_id: string; seed: number; frames: Frame[] };

const API = process.env.NEXT_PUBLIC_BACKEND_URL ?? "http://localhost:8000";
const empty: Frame = { frame_id: "waiting", replay_id: "none", theta: 0, eta: 0, tau: 0, target_lock_state: "SEARCHING", confidence: 0, timestamp: new Date(0).toISOString(), closing_speed_mps: 0, battery_percent: 100, gps: { lat: 0, lon: 0 }, threat_log: ["Waiting for synthetic telemetry"], line_of_sight_deg: 0, bearing_rate_dps: 0, predicted_intercept_point: { x: .5, y: .5 }, acceleration_vector: { x: 0, y: 0 }, lgmd_spike: 0 };

export default function Cockpit() {
  const [frame, setFrame] = useState<Frame>(empty);
  const [mode, setMode] = useState<"LIVE" | "DEMO">("LIVE");
  const [replay, setReplay] = useState<Replay | null>(null);
  const [replayIndex, setReplayIndex] = useState(0);
  const [trace, setTrace] = useState<number[]>([]);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const accept = useCallback((next: Frame) => { setFrame(next); setTrace((old) => [...old.slice(-28), next.lgmd_spike]); }, []);

  useEffect(() => {
    if (mode !== "LIVE") return;
    const source = new EventSource(`${API}/telemetry/stream`);
    source.onmessage = (event) => accept(JSON.parse(event.data) as Frame);
    source.onerror = () => source.close();
    return () => source.close();
  }, [accept, mode]);

  useEffect(() => {
    if (mode !== "DEMO" || !replay) return;
    const timer = window.setInterval(() => setReplayIndex((i) => (i + 1) % replay.frames.length), 650);
    return () => window.clearInterval(timer);
  }, [mode, replay]);

  useEffect(() => { if (mode === "DEMO" && replay) accept(replay.frames[replayIndex]); }, [accept, mode, replay, replayIndex]);

  useEffect(() => {
    let raf = 0; const canvas = canvasRef.current; if (!canvas) return;
    const context = canvas.getContext("2d"); if (!context) return;
    const draw = (time: number) => {
      const dpr = window.devicePixelRatio || 1; const w = canvas.clientWidth; const h = canvas.clientHeight;
      if (canvas.width !== w * dpr || canvas.height !== h * dpr) { canvas.width = w * dpr; canvas.height = h * dpr; }
      context.setTransform(dpr, 0, 0, dpr, 0, 0); context.fillStyle = "#071119"; context.fillRect(0, 0, w, h);
      for (let i = 0; i < 32; i++) { const x = (i * 47 + time / 24) % w; const y = (i * 29 + Math.sin(time / 800 + i) * 18 + h / 2) % h; context.fillStyle = i % 5 === 0 ? "#ffcb6b" : "#42d6c7"; context.globalAlpha = .25 + (i % 4) / 8; context.fillRect(x, y, 2, 2); }
      context.globalAlpha = 1; const size = 24 + frame.theta * 1.6; const cx = w * .52; const cy = h * .5;
      context.strokeStyle = frame.target_lock_state === "LOCKED" ? "#ff607c" : "#42d6c7"; context.lineWidth = 2; context.strokeRect(cx - size / 2, cy - size / 2, size, size);
      context.fillStyle = "#ff607c"; context.beginPath(); context.arc(w * frame.predicted_intercept_point.x, h * frame.predicted_intercept_point.y, 4, 0, Math.PI * 2); context.fill();
      context.strokeStyle = "#ffcb6b"; context.beginPath(); context.moveTo(cx, cy); context.lineTo(cx + frame.acceleration_vector.x * 55, cy + frame.acceleration_vector.y * 55); context.stroke();
      raf = requestAnimationFrame(draw);
    }; raf = requestAnimationFrame(draw); return () => cancelAnimationFrame(raf);
  }, [frame]);

  async function toggleDemo() {
    if (mode === "DEMO") { setMode("LIVE"); return; }
    const response = await fetch(`${API}/telemetry/snapshot`); const data = await response.json() as Replay;
    setReplay(data); setReplayIndex(0); setMode("DEMO");
  }

  return <main className="cockpit">
    <header><div><p className="eyebrow">SWATTERFLY / PHASE 1</p><h1>STITCH COCKPIT</h1></div><div className="status"><span className="dot" />{mode === "LIVE" ? "LIVE SYNTHETIC" : "DEMO REPLAY"}<button onClick={toggleDemo}>{mode === "LIVE" ? "DEMO REPLAY" : "RETURN LIVE"}</button></div></header>
    <div className="read-only">READ-ONLY OBSERVATION · ZERO FLIGHT AUTHORITY · NO COMMAND PATH</div>
    <section className="grid top"><article className="panel camera"><div className="panel-title">LIVE EVENT-CAM FEED <span>{frame.target_lock_state}</span></div><canvas ref={canvasRef} /><div className="camera-caption">canvas event dots · target lock box · guidance markers are explanatory only</div></article>
      <article className="panel ring-panel"><div className="panel-title">LOOMING RING</div><div className="ring" style={{ "--ring": `${Math.min(100, frame.theta * 2.2)}%` } as React.CSSProperties}><strong>{frame.theta.toFixed(1)}°</strong><small>THETA / ANGULAR SIZE</small></div><div className="countdown"><b>{frame.tau.toFixed(2)}s</b><span>TAU COUNTDOWN</span></div></article>
      <article className="panel meters"><div className="panel-title">ETA METER <span>0—1 / SYNTHETIC</span></div><div className="meter"><i style={{ width: `${frame.eta * 100}%` }} /></div><div className="meter-value">{frame.eta.toFixed(2)}</div><div className="panel-title spike-title">LGMD SPIKE TRACE <span>NEURON FIRING</span></div><div className="spikes">{trace.map((value, i) => <i key={`${i}-${value}`} style={{ height: `${Math.max(8, value * 100)}%` }} />)}</div></article></section>
    <section className="grid lower"><article className="panel guidance"><div className="panel-title">GUIDANCE OVERLAY <span>READ-ONLY</span></div><div className="overlay"><div className="los" /><div className="bearing">LOS {frame.line_of_sight_deg.toFixed(1)}° · RATE {frame.bearing_rate_dps.toFixed(1)}°/s</div><b className="intercept" style={{ left: `${frame.predicted_intercept_point.x * 100}%`, top: `${frame.predicted_intercept_point.y * 100}%` }}>×</b><div className="legend">— line of sight &nbsp; × predicted intercept &nbsp; → acceleration vector</div></div></article>
      <article className="panel telemetry"><div className="panel-title">TELEMETRY</div><div className="stats"><span>CLOSING SPEED<b>{frame.closing_speed_mps.toFixed(1)} m/s</b></span><span>BATTERY<b>{frame.battery_percent.toFixed(1)}%</b></span><span>GPS<b>{frame.gps.lat.toFixed(4)}, {frame.gps.lon.toFixed(4)}</b></span><span>TARGET CONFIDENCE<b>{(frame.confidence * 100).toFixed(0)}%</b></span></div><h3>THREAT LOG TIMESTAMPS</h3><ul>{frame.threat_log.map((item) => <li key={item}>{item}</li>)}</ul></article></section>
    <footer>FRAME {frame.frame_id} · REPLAY {frame.replay_id} · DETERMINISTIC SEED {replay?.seed ?? "—"} · {new Date(frame.timestamp).toISOString()}</footer>
  </main>;
}
