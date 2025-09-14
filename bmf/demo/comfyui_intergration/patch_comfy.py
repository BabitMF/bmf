# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 BabitMF

"""
Apply a minimal, non-invasive injection to ComfyUI's execution.py that imports
the Apache-licensed BMF runner and installs a runtime hook. This avoids bundling
any GPL code in our repository: users clone ComfyUI and run this script locally.

Safe to run multiple times; the script is idempotent.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


INJECTION_BLOCK = """
# === BMF_INTEGRATION_HOOK_START ===
try:
    from demo.comfyui_intergration import bmf_runner as _bmf_runner  # type: ignore
    _bmf_runner.install_bmf_engine_hook()  # type: ignore
except Exception as _bmf_exc:  # noqa: F841
    try:
        import logging  # type: ignore
        logging.warning(f"BMF hook install failed: {_bmf_exc}")
    except Exception:
        pass
# === BMF_INTEGRATION_HOOK_END ===
""".lstrip("\n")


def already_patched(text: str) -> bool:
    return "BMF_INTEGRATION_HOOK_START" in text


def apply_patch(comfy_path: Path) -> bool:
    exec_py = comfy_path / "execution.py"
    if not exec_py.exists():
        raise FileNotFoundError(f"execution.py not found under {comfy_path}")

    original = exec_py.read_text(encoding="utf-8")
    if already_patched(original):
        return False

    # Append the injection block at the end to avoid conflicts with upstream diffs
    updated = original.rstrip() + "\n\n" + INJECTION_BLOCK
    exec_py.write_text(updated, encoding="utf-8")
    return True


def main():
    parser = argparse.ArgumentParser(description="Inject BMF runtime hook into ComfyUI execution.py")
    parser.add_argument("--comfy-path", default="ComfyUI", help="Path to ComfyUI repo (directory containing execution.py)")
    parser.add_argument("--apply", action="store_true", help="Apply patch (default action if omitted)")
    args = parser.parse_args()

    comfy_dir = Path(args.comfy_path).resolve()
    changed = apply_patch(comfy_dir)
    if changed:
        print(f"[BMF] Patched {comfy_dir / 'execution.py'} with BMF hook block.")
    else:
        print("[BMF] ComfyUI execution.py already contains BMF hook block; skipping.")


if __name__ == "__main__":
    main()


