## BMF ↔ ComfyUI Integration

This demo integrates BMF with ComfyUI by swapping ComfyUI’s default Python execution engine with BMF’s C++ engine. It converts a ComfyUI prompt graph to a BMF graph and executes each ComfyUI node inside BMF via a Python bridge.

### Key features
- Drop‑in engine swap: run existing ComfyUI workflows without changing nodes
- Zero‑copy tensor bridge (hmp/DLPack) between BMF and PyTorch where possible
- Persistent loader cache across requests for heavy models (configurable size)
- Real‑time progress and preview forwarding to the ComfyUI frontend
- In‑process execution by default to reuse model memory for faster runs

## Requirements
- BMF built with CUDA and a Python version (>=3.12) compatible with ComfyUI

## Quick start (native)
1) Build BMF with CUDA (once):
```bash
cd /root/bmf
export CMAKE_ARGS="-DBMF_ENABLE_CUDA=ON"
./build.sh
```

2) Set up the ComfyUI integration and dependencies:
```bash
cd /root/bmf/output/demo/comfyui_intergration
./setup.sh
```

3) Run the integrated ComfyUI server (uses BMF engine):
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
./setup.sh
python run_bmf_comfy.py
```

## Repository layout
- `bridge.py`: Node runner and graph converter, zero‑copy bridges, loader cache
- `execution.py`: Replaces ComfyUI execution; forces BMF path and reports progress
- `run_bmf_comfy.py`: Launcher that sets import order and starts ComfyUI
- `setup.sh`: Clones ComfyUI and installs requirements for the chosen Python
- `ComfyUI/`: Vendored ComfyUI tree used by the demo launcher

## Notes
- This demo focuses on typical ComfyUI nodes. Exotic nodes may require additional handling.

## Licensing
- This integration demo combines BMF (Apache-2.0) with ComfyUI (GPL-3.0). The BMF core project remains licensed under Apache-2.0.
- File-level licenses in this directory:
  - `execution.py`: derivative of ComfyUI's `execution.py`; licensed under GPL-3.0. Portions Copyright (c) the ComfyUI contributors; modifications Copyright (c) 2025 BabitMF.
  - `bridge.py`, `run_bmf_comfy.py`, `setup.sh`, this `README.md`: licensed under Apache-2.0.
- Distribution guidance (non-legal advice): If you distribute this demo together with ComfyUI code or binaries that include it, you must comply with the GPL-3.0 terms. Using BMF core in commercial products is unaffected; keep this demo as an optional, separable component if you wish to avoid extending GPL obligations beyond the integration itself.
- License references:
  - ComfyUI: GPL-3.0 (`https://github.com/comfyanonymous/ComfyUI?tab=GPL-3.0-1-ov-file`)
  - BMF: Apache-2.0 (`https://github.com/BabitMF/bmf?tab=Apache-2.0-1-ov-file`)
