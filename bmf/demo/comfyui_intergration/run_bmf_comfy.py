# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 BabitMF
import sys
import os
import importlib.util

# Paths
integration_dir = os.path.dirname(os.path.abspath(__file__))
bmf_output_dir = os.path.abspath(os.path.join(integration_dir, '..', '..'))
comfyui_path = os.path.join(integration_dir, 'ComfyUI')

# Ensure import order: our integration -> bmf -> ComfyUI
if integration_dir not in sys.path:
    sys.path.insert(0, integration_dir)
if bmf_output_dir not in sys.path:
    sys.path.insert(1, bmf_output_dir)
if comfyui_path not in sys.path:
    sys.path.insert(2, comfyui_path)

# Import and run ComfyUI
os.chdir(comfyui_path)
sys.path.insert(0, comfyui_path)
from ComfyUI import main as comfy_main

# Install BMF runtime hook into ComfyUI's executor (no on-disk modification required)
try:
    import importlib
    execution_mod = importlib.import_module('execution')
    from demo.comfyui_intergration import bmf_runner
    bmf_runner.install_bmf_engine_hook(execution_mod)
    print("[BMF] Installed runtime hook into ComfyUI execution.PromptExecutor.execute")
except Exception as e:
    print(f"[BMF] WARNING: failed to install BMF runtime hook: {e}")

if __name__ == '__main__':
    event_loop, _, start_all = comfy_main.start_comfyui()
    try:
        event_loop.run_until_complete(start_all())
    except KeyboardInterrupt:
        pass
    finally:
        comfy_main.cleanup_temp()
