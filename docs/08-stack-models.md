# Stack and models

## NVIDIA and simulation

- **NVIDIA Cosmos-Transfer2.5** — sim-to-real augmentation; preserve depth and segmentation so geometry and labels remain meaningful.
- **Cosmos-Predict2.5** — alternative future-frame prediction path for forecasting experiments; it is not required for the first demo.
- **Isaac Lab** — physics simulation and reinforcement learning, run in controlled Nebius GPU bursts before any hardware transfer.

## Models and coding agents

- **Nemotron-3-Nano-4B** — optional onboard mission planner. It is described here as **32GB not Jetson-friendly**; use a GGUF-quantized form only if a measured memory/latency budget supports it. It must remain advisory behind deterministic gates.
- **GLM-5.3 + DeepSeek V4-Pro-0813** — coding agents on Nebius Token Factory. They assist implementation and analysis; they are not flight controllers.

## Edge and robotics stack

- **Jetson Orin Nano Super** — **$249, 67 TOPS, 102GB/s** as the intended edge target for the demo path.
- **Orin NX Super** — **157 TOPS** as a higher-headroom alternative.
- **Isaac ROS + TensorRT + Holoscan** — free stack for robotics integration, accelerated inference, and streaming dataflow where supported by the target hardware.
- **Tavily** — free **1000 credits/mo** for pre-mission research, subject to current account terms and rate limits.

## Design boundary

The onboard path should keep reflex detection, state estimation, and hard safety behavior local and bounded. Nebius is for training, hosting coding agents, and experiment orchestration. Cosmos is for data/simulation augmentation. Tavily is for a time-stamped mission brief. Any model, cloud result, or web-derived fact that is unavailable or stale must fail closed rather than silently becoming a flight command.
