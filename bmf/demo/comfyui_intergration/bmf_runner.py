# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 BabitMF

"""
Runtime hook and runner to execute ComfyUI prompts using BMF without bundling or
deriving from ComfyUI's GPL-licensed code. This module monkey-patches
ComfyUI's execution at runtime or via a tiny on-disk injection performed by
`patch_comfy.py`.

Usage options:
- Dynamic (recommended): import this module after ComfyUI's `execution` module
  is loaded and call `install_bmf_engine_hook(execution)`.
- On-disk injection: run `patch_comfy.py` after cloning ComfyUI to append a
  small import+hook stub at the end of `ComfyUI/execution.py`.

Controls:
- Environment variable `BMF_COMFY_FORCE` (default: "1"). When set to "1",
  the patched executor runs the BMF engine by default; when "0", it defers to
  ComfyUI's stock executor unless `extra_data.get("enable_bmf")` is truthy.
"""

from __future__ import annotations

import os
import time
import types
import logging
from typing import Any, Callable, Optional


def _should_use_bmf(extra_data: dict[str, Any] | None) -> bool:
    force_env = os.environ.get("BMF_COMFY_FORCE", "1").strip()
    if force_env == "1":
        return True
    if extra_data is None:
        return False
    return bool(extra_data.get("enable_bmf", False))


def _execute_with_bmf(self_obj: Any,
                      prompt: dict,
                      prompt_id: str,
                      extra_data: Optional[dict] = None,
                      execute_outputs: Optional[list] = None) -> None:
    """Execution path that converts the prompt to a BMF graph and runs it
    in the current process to maximize model cache reuse.

    This function intentionally does not depend on or reuse ComfyUI's execution
    code. It only calls ComfyUI's public modules to keep UI progress and events
    consistent.
    """
    import bmf
    #from bmf import BMF_TRACE_INIT, BMF_TRACE_DONE
    from bmf.python_sdk import Timestamp

    #logging.warning(f"[Debug BMF] _execute_with_bmf received extra_data: {extra_data}")

    # Bind client metadata to keep UI events wired correctly
    if extra_data and "client_id" in extra_data:
        self_obj.server.client_id = extra_data["client_id"]
    else:
        self_obj.server.client_id = None
    self_obj.server.last_prompt_id = prompt_id

    # Notify UI start and align with native fields
    self_obj.add_message("execution_start", {"prompt_id": prompt_id}, broadcast=False)
    # Ensure fields expected by main.prompt_worker
    try:
        if not hasattr(self_obj, "history_result") or self_obj.history_result is None:
            self_obj.history_result = {"outputs": {}, "meta": {}}
    except Exception:
        pass
    self_obj.success = False
    # Expose executor globally so bridge can accumulate UI outputs/meta
    try:
        from demo.comfyui_intergration.bridge import set_executor_instance
        set_executor_instance(self_obj)
    except Exception:
        pass

    #BMF_TRACE_INIT()

    # Initialize Comfy progress registry and hook
    try:
        from comfy_execution.graph import DynamicPrompt
        from comfy_execution.progress import (
            reset_progress_state,
            add_progress_handler,
            WebUIProgressHandler,
        )

        dynamic_prompt = DynamicPrompt(prompt)
        reset_progress_state(prompt_id, dynamic_prompt)
        add_progress_handler(WebUIProgressHandler(self_obj.server))
    except Exception:
        # Non-fatal; continue execution without progress integration
        pass

    # Convert the Comfy prompt graph to a BMF graph config
    from demo.comfyui_intergration.bridge import BmfWorkflowConverter

    # Enrich extra_data with prompt/dynprompt so V3 output nodes can embed metadata
    try:
        enriched_extra = dict(extra_data or {})
        enriched_extra.setdefault("prompt", prompt)
        try:
            enriched_extra.setdefault("dynprompt", dynamic_prompt)
        except Exception:
            pass
    except Exception:
        enriched_extra = extra_data
    #logging.warning(f"[Debug BMF] Calling BmfWorkflowConverter with extra_data: {enriched_extra}")
    converter = BmfWorkflowConverter(prompt, self_obj.server, enriched_extra)
    graph_config = converter.convert(execute_outputs or [])

    bmf_graph = bmf.graph()
    try:
        logging.info("[BMF] Running BMF graph for prompt_id=%s", prompt_id)
        returned_stream_names = bmf_graph.run_by_config(graph_config)
        if returned_stream_names:
            logging.info("[BMF] Polling output streams: %s", returned_stream_names)
            for stream_name in returned_stream_names:
                logging.info("[BMF] Polling stream: %s", stream_name)
                none_streak = 0
                while True:
                    pkt = bmf_graph.poll_packet(stream_name, True)
                    if not pkt or not pkt.defined():
                        none_streak += 1
                        if none_streak > 5:
                            break
                        time.sleep(0.1)
                        continue
                    none_streak = 0
                    if pkt.timestamp == Timestamp.EOF:
                        # Ensure all node progress bars complete in the UI
                        try:
                            from comfy_execution.progress import get_progress_state

                            reg = get_progress_state()
                            for nid in list(reg.nodes.keys()):
                                reg.finish_progress(nid)
                        except Exception:
                            pass
                        break
        logging.info("[BMF] BMF graph run finished for prompt_id=%s", prompt_id)
        self_obj.success = True
        # history_result may have been populated incrementally by bridge; keep if empty
        try:
            if not hasattr(self_obj, "history_result") or self_obj.history_result is None:
                self_obj.history_result = {"outputs": {}, "meta": {}}
        except Exception:
            pass
        self_obj.add_message("execution_success", {"prompt_id": prompt_id}, broadcast=False)
    finally:
        logging.info("[BMF] Closing BMF graph for prompt_id=%s", prompt_id)
        #BMF_TRACE_DONE()
        try:
            bmf_graph.close()
        except Exception:
            try:
                bmf_graph.force_close()
            except Exception:
                pass


def install_bmf_engine_hook(execution_module: Optional[types.ModuleType] = None) -> None:
    """Monkey-patch ComfyUI's PromptExecutor.execute to route into BMF.

    If `BMF_COMFY_FORCE=1` (default), BMF is always used. Otherwise, the
    original executor is used unless `extra_data['enable_bmf']` is truthy.

    This modifies behavior only at runtime and does not copy GPL code.
    """
    import importlib

    # Resolve ComfyUI's execution module if not provided
    if execution_module is None:
        try:
            execution_module = importlib.import_module("execution")
        except Exception as exc:
            raise RuntimeError(
                "Failed to import ComfyUI 'execution' module. Ensure ComfyUI is on sys.path."
            ) from exc

    PromptExecutor = getattr(execution_module, "PromptExecutor", None)
    if PromptExecutor is None:
        raise RuntimeError("ComfyUI execution.PromptExecutor not found; incompatible version?")

    original_execute: Optional[Callable[..., Any]] = getattr(PromptExecutor, "execute", None)
    if not callable(original_execute):
        raise RuntimeError("ComfyUI execution.PromptExecutor.execute not found; incompatible version?")

    # Avoid double-patching
    if getattr(PromptExecutor.execute, "__bmf_hook_installed__", False):
        return

    def _patched_execute(self, prompt, prompt_id, extra_data={}, execute_outputs=[]):
        try:
            if _should_use_bmf(extra_data):
                return _execute_with_bmf(self, prompt, prompt_id, extra_data, execute_outputs)
        except Exception as e:
            logging.error("[BMF] Falling back to original executor due to error: %s", e)
        return original_execute(self, prompt, prompt_id, extra_data, execute_outputs)

    # Mark function so we don't patch multiple times
    _patched_execute.__bmf_hook_installed__ = True  # type: ignore[attr-defined]

    setattr(PromptExecutor, "execute", _patched_execute)
    logging.info("[BMF] Installed BMF engine hook into ComfyUI PromptExecutor.execute")


__all__ = [
    "install_bmf_engine_hook",
]


