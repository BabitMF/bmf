## BMF ↔ ComfyUI Integration

### Overview

This integration replaces ComfyUI’s Python prompt executor with BMF’s high‑performance C++ graph runtime without vendoring any ComfyUI code. At startup, a tiny hook is installed into ComfyUI’s `execution.PromptExecutor.execute`. When a workflow runs, we:
- Convert the ComfyUI JSON workflow into a BMF `GraphConfig` in topological order.
- Map each ComfyUI node to a BMF Python module (`ComfyNodeRunner`) that instantiates the node class and calls its `execute`/`FUNCTION`.
- Bridge tensors zero‑copy between BMF (hmp/DLPack) and PyTorch so images/latents avoid memcpy.
- Forward progress and preview signals to the Web UI and accumulate `history_result` to match native behavior.

The hook is installed at runtime via `run_bmf_comfy.py` (no on‑disk patching).

### Key features
- Drop‑in engine swap: run existing ComfyUI workflows unchanged; GPL code stays in your local clone
- Zero‑copy tensor bridge: hmp/DLPack ↔ `torch.Tensor` for IMAGE/LATENT/MASK payloads
- Smart loader cache: LRU cache across requests for heavy loader nodes; re‑marks models in ComfyUI’s model manager to preserve VRAM reuse
- Native‑like UI integration: WebUI progress bars, optional live previews, `executing/executed` events, and `history_result` population
- In‑process execution: maximize model reuse; no subprocess boundary; automatic fallback to the stock executor on errors or when `BMF_COMFY_FORCE=0`
- Deterministic scheduling: framesync aligns multi‑input nodes; generator mode exposes terminal streams for polling

## Requirements
- BMF built with CUDA and a Python version (>=3.12) compatible with ComfyUI

## Quick start (native)
1) Build BMF with CUDA (once):
```bash
cd /root/bmf
export CMAKE_ARGS="-DBMF_ENABLE_CUDA=ON"
./build.sh
```

2) Set up the ComfyUI integration and dependencies (clones ComfyUI, installs deps):
```bash
cd /root/bmf/output/demo/comfyui_intergration
./setup.sh
```

3) Run the integrated ComfyUI server (uses BMF engine by default):
```bash
python run_bmf_comfy.py
```

4) Open http://localhost:8188 and execute your workflow as usual.

## Quick start (Docker)
If you prefer a container or you are on Windows via WSL2:

```bash
docker pull babitmf/bmf_runtime:latest

# Linux host example
docker run --name bmf --gpus all --network host \
  -v "/workspace" -w /root \
  babitmf/bmf_runtime:latest bash

# Windows + WSL2 example (exposes host CUDA libs into the container)
docker run --name bmf --gpus all --network host \
  -v /usr/lib/wsl/lib:/usr/lib/wsl/lib:ro \
  -e LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH \
  -v "/workspace:/root" \
  babitmf/bmf_runtime:latest bash
```

Inside the container:
```bash
cd /root/bmf
export CMAKE_ARGS="-DBMF_ENABLE_CUDA=ON"
./build.sh
cd /root/bmf/output/demo/comfyui_intergration
./setup.sh  # clones ComfyUI, installs deps
python run_bmf_comfy.py
```

## Repository layout
- `bridge.py`: Converts ComfyUI JSON to a BMF `GraphConfig` (topological sort, stream wiring). Implements `ComfyNodeRunner` to instantiate node classes, adapt inputs/outputs, perform zero‑copy tensor bridging, forward UI events, and cache loader outputs.
- `bmf_runner.py`: Runtime hook that replaces `PromptExecutor.execute` with `_execute_with_bmf`, sets up progress handlers, builds and runs the BMF graph, polls generator outputs, and ensures clean shutdown.
- `run_bmf_comfy.py`: Launcher that adds import paths, installs the runtime hook, and boots ComfyUI.
- `setup.sh`: Convenience script to clone ComfyUI and install requirements.
- `ComfyUI/`: Created by `setup.sh`; ComfyUI itself is not vendored here

## Notes
- This demo focuses on typical ComfyUI nodes. Exotic nodes may require additional handling.
- Controls: Environment variable `BMF_COMFY_FORCE` (default: "1"). When set to "1",
  the hooked executor runs the BMF engine by default; when "0", it defers to
  ComfyUI's stock executor unless `extra_data.get("enable_bmf")` is truthy.
- To run stock ComfyUI without BMF, use: `cd ComfyUI && python main.py`. Also, you can run our launcher but default to the stock engine with: `BMF_COMFY_FORCE=0 python run_bmf_comfy.py`.