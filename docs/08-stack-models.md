# Stack and models

## NVIDIA and simulation

- **Cosmos-Transfer2.5** and **Cosmos-Predict2.5** are no longer under active development and are in limited maintenance. The NVIDIA README states: “This repository is no longer under active development and will receive only limited maintenance updates.” The [NVIDIA Cosmos repository](https://github.com/nvidia/cosmos) and [Cosmos framework repository](https://github.com/NVIDIA/cosmos-framework) encourage migration to Cosmos 3.
- For the Phase 7 direction, prefer **Cosmos3-Nano (16B, H100-friendly)** through **vLLM-Omni** or **Diffusers**. Its OpenAI-compatible endpoint is `/v1/videos/sync`; depth, seg, edge, and wsn transfer hints are passed through `extra_params`. See the [Cosmos 3 model collection](https://huggingface.co/collections/nvidia/cosmos3).
- **Cosmos3-Edge (4B, Jetson)** currently lacks video-to-video transfer. **Cosmos 3 NIM** lacks transfer controls, so avoid both for Phase 7.
- Cosmos 3 uses the OpenMDW-1.1 license, which permits commercial and non-commercial use.
- Cosmos 3 is not yet on Nebius Token Factory. It was announced via Physical AI Workbench on AI Cloud in the [Nebius announcement](https://nebius.com/blog/posts/run-physical-ai-workflows-not-glue-code).
- Retain **Cosmos-Transfer2.5** only as a short-term fallback while the Cosmos 3 path is validated.
- Generated frames are not pixel-aligned depth/segmentation ground truth. Regenerate labels through Isaac Sim before using them as training or evaluation labels.

## Nebius Token Factory configuration

Use the following configuration for model calls:

- `base_url`: `https://api.tokenfactory.nebius.com/v1/`
- environment variable: `NEBIUS_API_KEY` — not bare `OPENAI_API_KEY`
- OpenAI-SDK tools, including IsaacLabEureka, must be explicitly patched to that Token Factory base URL.

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
