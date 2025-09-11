import sys
import os
import importlib.util

# Paths
integration_dir = os.path.dirname(os.path.abspath(__file__))
bmf_output_dir = os.path.abspath(os.path.join(integration_dir, '..', '..'))
comfyui_path = os.path.join(integration_dir, 'ComfyUI')

# Ensure import order: our execution.py -> bmf -> ComfyUI
if integration_dir not in sys.path:
    sys.path.insert(0, integration_dir)
if bmf_output_dir not in sys.path:
    sys.path.insert(1, bmf_output_dir)
if comfyui_path not in sys.path:
    sys.path.insert(2, comfyui_path)

# Preload our execution.py so any `import execution` resolves here
exec_path = os.path.join(integration_dir, 'execution.py')
if os.path.exists(exec_path):
    spec = importlib.util.spec_from_file_location('execution', exec_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules.pop('execution', None)
    sys.modules['execution'] = mod
    spec.loader.exec_module(mod)
    print(f"[BMF] execution: {mod.__file__}")
else:
    print("[BMF] WARNING: integration execution.py not found; falling back to ComfyUI's execution.py")

# Import and run ComfyUI
os.chdir(comfyui_path)
sys.path.insert(0, comfyui_path)
from ComfyUI import main as comfy_main

if __name__ == '__main__':
    event_loop, _, start_all = comfy_main.start_comfyui()
    try:
        event_loop.run_until_complete(start_all())
    except KeyboardInterrupt:
        pass
    finally:
        comfy_main.cleanup_temp()
