# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 BabitMF

import bmf
from bmf import Module, Task, Packet, Timestamp, ProcessResult
import sys
import os
import torch
import torch.utils.dlpack as torch_dlpack
import asyncio
import inspect
from bmf.builder.graph_config import GraphConfig, NodeConfig, StreamConfig, ModuleConfig
import logging
from types import SimpleNamespace
try:
    import bmf.hmp as hmp
except Exception as e:
    hmp = None
from comfy.cli_args import args

# Add ComfyUI to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'ComfyUI')))
import nodes
from collections import OrderedDict
from comfy_execution.utils import CurrentNodeContext

# Persistent LRU cache for loader node outputs to enable cross-request reuse with bounded memory
class _LRUCache:
    def __init__(self, capacity: int = 16):
        self.capacity = capacity
        self.data = OrderedDict()

    def get(self, key):
        if key not in self.data:
            return None
        value = self.data.pop(key)
        self.data[key] = value
        return value

    def set(self, key, value):
        if key in self.data:
            self.data.pop(key)
        self.data[key] = value
        while len(self.data) > self.capacity:
            # Drop least recently used entry to release references
            self.data.popitem(last=False)

_CACHE_CAP = int(os.environ.get('BMF_COMFY_LOADER_CACHE_SIZE', '16'))
PERSISTENT_NODE_CACHE = _LRUCache(capacity=max(1, _CACHE_CAP))

# Global handle to ComfyUI server for progress/preview and executing updates
GLOBAL_SERVER_INSTANCE = None
# Global handle to current PromptExecutor instance to align with native caches/history
GLOBAL_EXECUTOR_INSTANCE = None

def set_server_instance(server):
    global GLOBAL_SERVER_INSTANCE
    GLOBAL_SERVER_INSTANCE = server

def set_executor_instance(executor):
    global GLOBAL_EXECUTOR_INSTANCE
    GLOBAL_EXECUTOR_INSTANCE = executor


class ComfyNodeRunner(Module):
    def __init__(self, node_id=None, option=None):
        super().__init__(node_id, option)
        self.option = option
        #logging.warning(f"[Debug BMF] ComfyNodeRunner.__init__ received option for node {option.get('comfy_node_id', '')}: {option}")
        
        self.class_type = option['class_type']
        self.comfy_node_class = nodes.NODE_CLASS_MAPPINGS[self.class_type]
        self.comfy_node_instance = self.comfy_node_class()
        
        self.widget_inputs = {}
        self.link_inputs_info = {} 
        self.input_type_map = {}
        self.extra_data = option.get('extra_data')

        # Separate widget inputs from linked inputs
        if 'inputs' in self.option:
            input_idx = 0
            # Get the canonical input order to correctly map stream index to input name
            input_order = self._get_input_order(self.class_type)
            all_inputs = self.option.get('inputs', {})

            for name in input_order:
                if name in all_inputs:
                    value = all_inputs[name]
                    if isinstance(value, list):
                        self.link_inputs_info[input_idx] = name
                        input_idx += 1
            
            for name, value in all_inputs.items():
                 if not isinstance(value, list):
                    self.widget_inputs[name] = value

        # Build expected type map for inputs (IMAGE/LATENT/MASK/...)
        self.input_type_map = self._build_input_type_map()

        # Identify if this node is a cacheable loader (model/clip/vae/controlnet/etc.)
        self.is_loader_node = self._is_cacheable_loader(self.class_type)

        # Inject comfy_api hidden tokens if this is a comfy_api.latest style node
        try:
            self._inject_comfy_api_hidden()
        except Exception:
            pass

    def _get_input_order(self, class_type):
        """Gets the canonical input order from the node's class definition."""
        try:
            node_class = nodes.NODE_CLASS_MAPPINGS[class_type]
            input_types = node_class.INPUT_TYPES()
            required = list(input_types.get('required', {}).keys())
            optional = list(input_types.get('optional', {}).keys())
            return required + optional
        except Exception:
            return []

    def _build_input_type_map(self):
        """Infer expected ComfyUI type name for each input (best-effort)."""
        type_map = {}
        try:
            input_types = self.comfy_node_class.INPUT_TYPES()
            for section in ('required', 'optional'):
                if section in input_types:
                    for name, spec in input_types[section].items():
                        typ = None
                        if isinstance(spec, tuple) and len(spec) > 0:
                            base = spec[0]
                        else:
                            base = spec
                        if isinstance(base, str):
                            typ = base
                        else:
                            # IO.* or other enums/classes
                            typ = getattr(base, 'name', str(base))
                        type_map[name] = typ
        except Exception:
            pass
        return type_map

    def _torch_from_hmp(self, hmp_tensor):
        """Convert hmp.Tensor -> torch.Tensor via DLPack (zero-copy)."""
        if hmp is None:
            raise RuntimeError("bmf.hmp is not available")
        # Prefer direct torch() bridge if available
        to_torch = getattr(hmp_tensor, 'torch', None)
        if callable(to_torch):
            return to_torch()
        # Fallback to DLPack
        cap = hmp_tensor.__dlpack__(1)
        return torch_dlpack.from_dlpack(cap)

    def _hmp_from_torch(self, torch_tensor):
        """Convert torch.Tensor -> hmp.Tensor via DLPack (zero-copy)."""
        if hmp is None:
            raise RuntimeError("bmf.hmp is not available")
        # Prefer direct from_torch() if available
        if hasattr(hmp, 'from_torch'):
            return hmp.from_torch(torch_tensor)
        # Fallback to DLPack by passing tensor object (not capsule)
        return hmp.from_dlpack(torch_tensor)

    def _is_cacheable_loader(self, class_type: str) -> bool:
        """Return True if this class type loads persistent heavy models."""
        return class_type in {
            'CheckpointLoader', 'CheckpointLoaderSimple', 'unCLIPCheckpointLoader',
            'DiffusersLoader', 'VAELoader', 'CLIPLoader', 'DualCLIPLoader',
            'CLIPVisionLoader', 'ControlNetLoader', 'DiffControlNetLoader',
            'UNETLoader', 'StyleModelLoader', 'GLIGENLoader'
        }

    def _is_model_mutating_node(self, class_type: str) -> bool:
        """Nodes that patch/modify model weights or routing (should run outside inference_mode)."""
        return class_type in {
            'LoraLoader', 'LoraLoaderModelOnly', 'StyleModelApply', 'CLIPSetLastLayer'
        }

    def _make_cache_key(self) -> str:
        """Make a stable cache key from class type and widget inputs (sorted)."""
        try:
            import json
            key_inputs = {k: self.widget_inputs.get(k) for k in sorted(self.widget_inputs.keys())}
            return f"{self.class_type}|{json.dumps(key_inputs, sort_keys=True, default=str)}"
        except Exception:
            return f"{self.class_type}|{str(sorted(self.widget_inputs.items()))}"

    def _collect_models_from_result(self, results):
        """Best-effort to extract models to mark as used in Comfy's cache.
        Returns a list of model-like objects accepted by comfy.model_management.load_models_gpu
        """
        models = []
        def try_add(x):
            if x is None:
                return
            # Prefer ModelPatcher-like objects (have patching APIs)
            if hasattr(x, 'model_patches_to') and hasattr(x, 'patch_model'):
                models.append(x)
                return
            # Fallback: raw .model attribute
            m = getattr(x, 'model', None)
            if m is not None:
                models.append(m)
                return
        if isinstance(results, (list, tuple)):
            for x in results:
                try_add(x)
        else:
            try_add(results)
        
        print(f"[Debug BMF] ComfyNodeRunner {self.option.get('comfy_node_id', '')} _collect_models_from_result: collected {len(models)} models.")
        for i, m in enumerate(models):
            print(f"[Debug BMF]   model {i}: {type(m)}")
            
        return models

    def close(self):
        """Release strong references held by this module to encourage GC/VRAM release.
        Avoid global unloads here; the executor will manage model memory.
        """
        try:
            # Optional per-node cleanup hook
            if hasattr(self.comfy_node_instance, 'cleanup') and callable(self.comfy_node_instance.cleanup):
                try:
                    self.comfy_node_instance.cleanup()
                except Exception:
                    pass
        except Exception:
            pass
        # Drop references so tensors/modules can be GC'd
        self.comfy_node_instance = None
        self.widget_inputs = None
        self.link_inputs_info = None
        self.input_type_map = None
        return 0

    def _adapt_input_for_comfy(self, input_name, t):
        """Adapt a torch tensor into the exact structure Comfy expects for this input."""
        expected = self.input_type_map.get(input_name, None)
        # Handle our zero-copy latent wrapper
        if expected == 'LATENT' and isinstance(t, tuple) and len(t) >= 3 and t[0] == 'LATENT_ZC':
            meta = dict(t[1]) if isinstance(t[1], dict) else {}
            samples_hmp = t[2]
            noise_hmp = t[3] if len(t) > 3 else None
            samples_torch = self._torch_from_hmp(samples_hmp)
            meta['samples'] = samples_torch
            if noise_hmp is not None:
                meta['noise_mask'] = self._torch_from_hmp(noise_hmp)
            return meta
        # LATENT expects a dict with key 'samples' if given raw tensor
        if expected == 'LATENT' and isinstance(t, torch.Tensor):
            return {"samples": t}
        # IMAGE expects NHWC float tensor in [0,1]
        if expected == 'IMAGE' and isinstance(t, torch.Tensor):
            if t.dim() == 4 and t.shape[1] in (1, 3, 4) and t.shape[-1] not in (1, 3, 4):
                # NCHW -> NHWC using a view (no copy)
                t = t.permute(0, 2, 3, 1)
            return t
        # MASK can be [H,W] or [B,H,W] (leave as is)
        return t

    def _packet_from_output(self, obj, expected_type=None):
        """Wrap node output into a Packet using zero-copy when possible.
        - torch.Tensor -> hmp.Tensor inside Packet
        - dict with 'samples' tensor -> Packet of hmp.Tensor (samples)
        Others: fall back to bytes (non-zero-copy), but log once.
        """
        try:
            # Never emit a null packet; wrap None explicitly
            if obj is None:
                return Packet(("PYOBJ", None))
            # Optional preview aligned with ComfyUI: only for IMAGE outputs and opt-in via env
            # ComfyUI samplers already emit previews through latent_preview + global hook.
            # Here we avoid injecting previews for non-image outputs (e.g., LATENT) to match native behavior.
            try:
                if expected_type == 'IMAGE' and os.environ.get('BMF_COMFY_PREVIEW_IMAGES', '0') == '1':
                    from comfy_execution.progress import get_progress_state
                    from PIL import Image
                    import numpy as np
                    nid = str(self.option.get('comfy_node_id', ''))
                    preview_img = None
                    t = obj if isinstance(obj, torch.Tensor) else None
                    if t is not None and t.dim() >= 3:
                        t0 = t[0] if t.dim() == 4 else t
                        if t0.shape[0] in (1, 3, 4):
                            arr = (t0.detach().float().clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255.0).astype('uint8')
                        else:
                            arr = (t0.detach().float().clamp(0, 1).cpu().numpy() * 255.0).astype('uint8')
                        preview_img = Image.fromarray(arr)
                    if preview_img is not None and nid:
                        reg = get_progress_state()
                        image_tuple = ("JPEG", preview_img, getattr(args, 'preview_size', 512))
                        reg.update_progress(nid, reg.ensure_entry(nid)["value"], reg.ensure_entry(nid)["max"], image_tuple)
            except Exception:
                pass
            if isinstance(obj, torch.Tensor):
                return Packet(self._hmp_from_torch(obj))
            if isinstance(obj, dict) and 'samples' in obj and isinstance(obj['samples'], torch.Tensor):
                meta = dict(obj)
                samples = meta.pop('samples')
                noise = None
                if 'noise_mask' in meta and isinstance(meta['noise_mask'], torch.Tensor):
                    noise = meta.pop('noise_mask')
                tup = ('LATENT_ZC', meta, self._hmp_from_torch(samples), self._hmp_from_torch(noise) if isinstance(noise, torch.Tensor) else None)
                return Packet(tup)
        except Exception as e:
            logging.debug(f"Zero-copy packet wrap failed: {e}")
        # Fallbacks:
        # - If dict (may contain tensors), wrap to avoid JsonParam conversion
        if isinstance(obj, dict):
            return Packet(("PYOBJ", obj))
        # - Store Python object pointer directly (no serialization). Avoid bare None
        return Packet(("PYOBJ", obj))

    def process(self, task):
        # Progress start/event moved to actual execution time to avoid false-running states
        # Handle EOF: if any input stream is finished, we propagate EOF and finish this node.
        if task.get_inputs():
            is_eof = False
            for input_id, input_queue in task.get_inputs().items():
                if not input_queue.empty() and input_queue.front().timestamp == Timestamp.EOF:
                    is_eof = True
                    break
            if is_eof:
                for output_id, output_queue in task.get_outputs().items():
                    output_queue.put(Packet.generate_eof_packet())
                task.set_timestamp(Timestamp.DONE)
                # Finish progress only if it was started before
                try:
                    from comfy_execution.progress import get_progress_state
                    nid = str(self.option.get('comfy_node_id', ''))
                    reg = get_progress_state()
                    if nid and hasattr(reg, 'nodes') and (nid in reg.nodes):
                        reg.finish_progress(nid)
                except Exception:
                    pass
                return ProcessResult.OK

        kwargs = self.widget_inputs.copy()
        
        # Check if all inputs for this task are ready
        all_inputs_ready = True
        for input_id, input_queue in task.get_inputs().items():
            if input_queue.empty():
                all_inputs_ready = False
                break
        
        if not all_inputs_ready:
            # Not all inputs are ready, wait for more packets
            return ProcessResult.OK

        # Process inputs
        for i, (input_id, input_queue) in enumerate(task.get_inputs().items()):
            pkt = input_queue.get()
            # We only process data packets here. EOF is handled above.
            if pkt.timestamp != Timestamp.EOF:
                input_name = self.link_inputs_info[i]
                # Prefer zero-copy path via hmp.Tensor -> torch.Tensor
                value_set = False
                expected = self.input_type_map.get(input_name)
                heavy_types = {"IMAGE", "LATENT", "MASK"}
                if hmp is not None and expected in heavy_types:
                    try:
                        h = pkt.get(hmp.Tensor)
                        if h is not None:
                            t = self._torch_from_hmp(h)
                            kwargs[input_name] = self._adapt_input_for_comfy(input_name, t)
                            value_set = True
                    except Exception as e:
                        logging.debug(f"Zero-copy input decode failed for {input_name}: {e}")
                if not value_set:
                    # Fallback: get stored Python object directly (pointer, no copy)
                    py_obj = pkt.get(None)
                    # Guard against null/invalid payloads: treat as None
                    if py_obj is None:
                        kwargs[input_name] = None
                        continue
                    if isinstance(py_obj, tuple) and len(py_obj) >= 2 and py_obj[0] == 'PYOBJ':
                        py_obj = py_obj[1]
                    if py_obj is None:
                        # Leave as None to avoid raising and emitting null packets downstream
                        kwargs[input_name] = None
                        continue
                    # If it's our zero-copy latent wrapper, adapt to Comfy format
                    if isinstance(py_obj, tuple) and len(py_obj) >= 3 and py_obj[0] == 'LATENT_ZC':
                        kwargs[input_name] = self._adapt_input_for_comfy(input_name, py_obj)
                    # If it's hmp.Tensor carried as PythonObject
                    elif hmp is not None and isinstance(py_obj, hmp.Tensor):
                        t = self._torch_from_hmp(py_obj)
                        kwargs[input_name] = self._adapt_input_for_comfy(input_name, t)
                    else:
                        kwargs[input_name] = py_obj

        # Add hidden inputs from extra_data, mimicking ComfyUI's execution logic
        if self.extra_data is not None and hasattr(self.comfy_node_class, 'INPUT_TYPES'):
            #logging.warning(f"[Debug BMF] extra_data for node {self.option.get('comfy_node_id', '')}: {self.extra_data}")
            try:
                class_inputs = self.comfy_node_class.INPUT_TYPES()
                if "hidden" in class_inputs:
                    hidden_map = class_inputs["hidden"]
                    for key, value in hidden_map.items():
                        if value == "AUTH_TOKEN_COMFY_ORG":
                            kwargs[key] = self.extra_data.get("auth_token_comfy_org")
                        elif value == "API_KEY_COMFY_ORG":
                            kwargs[key] = self.extra_data.get("comfy_api_key")
                        elif value == "UNIQUE_ID":
                            kwargs[key] = str(self.option.get('comfy_node_id'))
                        elif value == "EXTRA_PNGINFO":
                            kwargs[key] = self.extra_data.get('extra_pnginfo')
                        elif value == "PROMPT":
                            pass # prompt is already part of the workflow
            except Exception:
                pass

        function_name = getattr(self.comfy_node_instance, 'FUNCTION', 'execute')
        execute_func = getattr(self.comfy_node_instance, function_name)

        logging.warning(f"[Debug BMF] kwargs keys for node {self.option.get('comfy_node_id', '')}: {kwargs.keys()}")

        # Try to reuse cached outputs for loader nodes
        cache_key = None
        results = None
        if self.is_loader_node:
            cache_key = self._make_cache_key()
            cached = PERSISTENT_NODE_CACHE.get(cache_key)
            if cached is not None:
                results = cached
                # Touch the underlying model loaders so Comfy marks them as currently used
                try:
                    import comfy.model_management as mm
                    models = self._collect_models_from_result(results)
                    if models:
                        print(f"[Debug BMF] ComfyNodeRunner {self.option.get('comfy_node_id', '')} calling load_models_gpu from cache path (force_full_load).")
                        mm.load_models_gpu(models)
                except Exception as e:
                    print(f"[Debug BMF] Error in load_models_gpu: {e}")
                    pass

        if results is None:
            # Filter kwargs to match the signature of the node's execution function.
            # This prevents TypeErrors for nodes that don't accept arbitrary (**kwargs).
            sig = inspect.signature(execute_func)
            params = sig.parameters
            has_var_keyword = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())

            if not has_var_keyword:
                filtered_kwargs = {k: v for k, v in kwargs.items() if k in params}
            else:
                filtered_kwargs = kwargs

            def do_execute():
                # Handle both sync and async node execution functions
                if inspect.iscoroutinefunction(execute_func):
                    return asyncio.run(execute_func(**filtered_kwargs))
                else:
                    return execute_func(**filtered_kwargs)

            # Loaders and model-mutating nodes: run OUTSIDE inference_mode to avoid creating inference tensors as params
            use_inference = not (self.is_loader_node or self._is_model_mutating_node(self.class_type))
            # Set executing context so Comfy's global progress hook can resolve node/prompt ids
            ctx_node_id = str(self.option.get('comfy_node_id', ''))
            prompt_id = None
            try:
                # Use the server's last prompt id if available
                prompt_id = getattr(GLOBAL_SERVER_INSTANCE, 'last_prompt_id', None)
            except Exception:
                prompt_id = None
            # Announce executing and mark progress start now that inputs are ready and we're about to run
            try:
                server = GLOBAL_SERVER_INSTANCE
                if server is not None and ctx_node_id:
                    server.last_node_id = ctx_node_id
                    if server.client_id is not None:
                        server.send_sync("executing", {"node": ctx_node_id, "display_node": ctx_node_id, "prompt_id": getattr(server, 'last_prompt_id', None)}, server.client_id)
                from comfy_execution.progress import get_progress_state
                if ctx_node_id:
                    get_progress_state().start_progress(ctx_node_id)
            except Exception:
                pass

            if use_inference:
                print(f"[Debug BMF] Node {self.class_type} ({self.option.get('comfy_node_id','')}) running under inference_mode", flush=True)
                with torch.inference_mode():
                    with CurrentNodeContext(prompt_id or '', ctx_node_id or '', None):
                        results = do_execute()
            else:
                print(f"[Debug BMF] Node {self.class_type} ({self.option.get('comfy_node_id','')}) running outside inference_mode", flush=True)
                with CurrentNodeContext(prompt_id or '', ctx_node_id or '', None):
                    results = do_execute()
            # If result is a V3 NodeOutput, unwrap args and capture UI for forwarding
            nodeoutput_ui = None
            try:
                from comfy_api.latest import io as comfy_io  # type: ignore
                NodeOutput = getattr(comfy_io, "NodeOutput", None)
            except Exception:
                NodeOutput = None
            if NodeOutput is not None and isinstance(results, NodeOutput):
                try:
                    nodeoutput_ui = getattr(results, "ui", None)
                except Exception:
                    nodeoutput_ui = None
                try:
                    # NodeOutput.result returns tuple or None
                    results = results.result
                except Exception:
                    pass
            # Forward UI output to frontend if provided by this node (e.g., SaveImage / SaveAudio / SaveVideo)
            try:
                if nodeoutput_ui is not None:
                    server = GLOBAL_SERVER_INSTANCE
                    node_id = str(self.option.get('comfy_node_id', ''))
                    display_node = node_id
                    ui_payload = None
                    try:
                        ui_payload = nodeoutput_ui.as_dict() if hasattr(nodeoutput_ui, 'as_dict') else nodeoutput_ui
                    except Exception:
                        ui_payload = nodeoutput_ui
                    if server is not None and server.client_id is not None:
                        server.send_sync("executed", {"node": node_id, "display_node": display_node, "output": ui_payload, "prompt_id": getattr(server, 'last_prompt_id', None)}, server.client_id)
                    try:
                        exec_inst = GLOBAL_EXECUTOR_INSTANCE
                        if exec_inst is not None:
                            if not hasattr(exec_inst, 'history_result') or exec_inst.history_result is None:
                                exec_inst.history_result = {"outputs": {}, "meta": {}}
                            outputs_map = exec_inst.history_result.setdefault("outputs", {})
                            meta_map = exec_inst.history_result.setdefault("meta", {})
                            outputs_map[node_id] = ui_payload
                            meta_map[node_id] = {
                                "node_id": node_id,
                                "display_node": display_node,
                                "parent_node": None,
                                "real_node_id": node_id,
                            }
                    except Exception:
                        pass
                elif isinstance(results, dict) and 'ui' in results:
                    server = GLOBAL_SERVER_INSTANCE
                    node_id = str(self.option.get('comfy_node_id', ''))
                    display_node = node_id
                    if server is not None and server.client_id is not None:
                        server.send_sync("executed", {"node": node_id, "display_node": display_node, "output": results['ui'], "prompt_id": getattr(server, 'last_prompt_id', None)}, server.client_id)
                    # Align with native: accumulate UI outputs into executor.history_result
                    try:
                        exec_inst = GLOBAL_EXECUTOR_INSTANCE
                        if exec_inst is not None:
                            if not hasattr(exec_inst, 'history_result') or exec_inst.history_result is None:
                                exec_inst.history_result = {"outputs": {}, "meta": {}}
                            outputs_map = exec_inst.history_result.setdefault("outputs", {})
                            meta_map = exec_inst.history_result.setdefault("meta", {})
                            outputs_map[node_id] = results['ui']
                            meta_map[node_id] = {
                                "node_id": node_id,
                                "display_node": display_node,
                                "parent_node": None,
                                "real_node_id": node_id,
                            }
                    except Exception:
                        pass
            except Exception:
                pass
            # Store outputs for cacheable loader nodes
            if self.is_loader_node and cache_key is not None:
                PERSISTENT_NODE_CACHE.set(cache_key, results)

        # Wrap outputs in BMF packets and send
        expected_outputs = getattr(self.comfy_node_class, 'RETURN_TYPES', ())
        expected_count = len(expected_outputs) if isinstance(expected_outputs, (tuple, list)) else 0

        # Normalize results into a list of length expected_count
        out_values = []
        if results is None:
            out_values = [None] * expected_count
        elif isinstance(results, tuple) or isinstance(results, list):
            out_values = list(results)
        elif isinstance(results, dict) and ('result' in results):
            r = results['result']
            if isinstance(r, (tuple, list)):
                out_values = list(r)
            else:
                out_values = [r]
        else:
            # Single object
            out_values = [results]

        if expected_count > 0:
            if len(out_values) < expected_count:
                out_values.extend([None] * (expected_count - len(out_values)))
            elif len(out_values) > expected_count:
                out_values = out_values[:expected_count]

        # Emit packets for each declared output index
        for i, output_queue in task.get_outputs().items():
            value = out_values[i] if i < len(out_values) else None
            expected_type = None
            try:
                expected_type = expected_outputs[i] if i < len(expected_outputs) else None
            except Exception:
                expected_type = None
            out_pkt = self._packet_from_output(value, expected_type=expected_type)
            out_pkt.timestamp = task.timestamp
            output_queue.put(out_pkt)

        # Source nodes (no inputs) send EOF after their single execution
        if not task.get_inputs():
            for output_id, output_queue in task.get_outputs().items():
                output_queue.put(Packet.generate_eof_packet())
            task.set_timestamp(Timestamp.DONE)

        # Mark node as finished in progress registry (only if previously started)
        try:
            from comfy_execution.progress import get_progress_state
            nid = str(self.option.get('comfy_node_id', ''))
            reg = get_progress_state()
            if nid and hasattr(reg, 'nodes') and (nid in reg.nodes):
                reg.finish_progress(nid)
        except Exception:
            pass

        return ProcessResult.OK

    def _inject_comfy_api_hidden(self):
        """For comfy_api.latest.io based nodes, emulate ComfyUI's hidden context by
        setting class-level `hidden` attributes expected by their execute methods.
        """
        try:
            # Lazy import to avoid hard dependency if comfy_api is absent
            from comfy_api.latest import io as comfy_io  # type: ignore
        except Exception:
            return
        try:
            if hasattr(self.comfy_node_class, "define_schema") and isinstance(self.comfy_node_class, type) and issubclass(self.comfy_node_class, getattr(comfy_io, "ComfyNode", object)):
                tokens = self.extra_data or {}
                hidden_obj = SimpleNamespace(
                    auth_token_comfy_org=tokens.get("auth_token_comfy_org"),
                    api_key_comfy_org=tokens.get("comfy_api_key"),
                    unique_id=str(self.option.get('comfy_node_id')),
                    extra_pnginfo=tokens.get("extra_pnginfo"),
                    prompt=tokens.get("prompt", None),
                    dynprompt=tokens.get("dynprompt", None),
                )
                setattr(self.comfy_node_class, "hidden", hidden_obj)
        except Exception:
            # Best-effort; if injection fails, let execution proceed and raise naturally
            pass

class BmfWorkflowConverter:
    def __init__(self, comfy_workflow, server_instance, extra_data=None):
        self.comfy_workflow = comfy_workflow
        self.server = server_instance
        self.extra_data = extra_data
        #logging.warning(f"[Debug BMF] BmfWorkflowConverter.__init__ received extra_data: {self.extra_data}")
        self.graph_config = GraphConfig()
        self.graph_config.set_option({"dump_graph": 1, "graph_name": "comfy_to_bmf"})
        # Expose server globally so modules can send executing/progress/preview events
        try:
            set_server_instance(server_instance)
        except Exception:
            pass

    def _get_input_order(self, class_type):
        """Gets the canonical input order from the node's class definition."""
        try:
            node_class = nodes.NODE_CLASS_MAPPINGS[class_type]
            input_types = node_class.INPUT_TYPES()
            required = list(input_types.get('required', {}).keys())
            optional = list(input_types.get('optional', {}).keys())
            return required + optional
        except Exception:
            return []

    def convert(self, execute_outputs=[]):
        # 1. Build dependency graph for topological sort
        in_degree = {node_id: 0 for node_id in self.comfy_workflow}
        adj = {node_id: [] for node_id in self.comfy_workflow}

        for node_id, node_info in self.comfy_workflow.items():
            if 'inputs' in node_info:
                for _, value in node_info['inputs'].items():
                    if isinstance(value, list):
                        from_node_id = str(value[0])
                        if from_node_id in self.comfy_workflow:
                            in_degree[node_id] += 1
                            adj[from_node_id].append(node_id)
        
        # 2. Initialize queue with source nodes
        queue = [node_id for node_id, degree in in_degree.items() if degree == 0]
        
        # 3. Process nodes in topological order to build GraphConfig
        processed_count = 0
        node_outputs = {} # map (comfy_node_id, port) -> stream_identifier
        stream_id_counter = 0

        while queue:
            node_id = queue.pop(0)
            node_info = self.comfy_workflow[node_id]
            class_type = node_info['class_type']
            
            node_config = NodeConfig()
            node_config.set_id(int(node_id))

            option = {
                "class_type": class_type,
                "inputs": node_info['inputs'],
                "comfy_node_id": node_id,
                "extra_data": self.extra_data
            }
            node_config.set_option(option)

            module_info_dict = {
                "name": 'ComfyNodeRunner',
                "type": "python",
                "path": "",
                "entry": 'demo.comfyui_intergration.bridge:ComfyNodeRunner'
            }
            node_config.set_module_info(ModuleConfig(module_info_dict))

            # Connect inputs by finding upstream stream identifiers
            if 'inputs' in node_info:
                links = {k: v for k, v in node_info['inputs'].items() if isinstance(v, list)}
                input_order = self._get_input_order(class_type)
                input_link_count = 0
                for input_name in input_order:
                    if input_name in links:
                        link_val = links[input_name]
                        from_node_id, from_port = str(link_val[0]), link_val[1]
                        stream_identifier = node_outputs.get((from_node_id, from_port))
                        if stream_identifier:
                            stream_conf = StreamConfig({"identifier": stream_identifier, "stream_alias": ""})
                            node_config.add_input_stream(stream_conf)
                            input_link_count += 1

            # Select input manager strategy
            # - framesync for nodes requiring multiple inputs ready together
            # - default for single-input or source nodes
            if node_config.get_input_streams() and len(node_config.get_input_streams()) >= 2:
                node_config.set_input_manager('framesync')
            else:
                node_config.set_input_manager('default')

            # Create output stream configs for the current node
            output_count = len(nodes.NODE_CLASS_MAPPINGS[class_type].RETURN_TYPES)
            for i in range(output_count):
                stream_identifier = f"stream_{node_id}_{i}_{stream_id_counter}"
                stream_id_counter += 1
                stream_conf = StreamConfig({"identifier": stream_identifier, "stream_alias": ""})
                node_config.add_output_stream(stream_conf)
                node_outputs[(node_id, i)] = stream_identifier

            self.graph_config.add_node_config(node_config)
            
            processed_count += 1
            
            # Update neighbors
            for neighbor_id in adj[node_id]:
                in_degree[neighbor_id] -= 1
                if in_degree[neighbor_id] == 0:
                    queue.append(neighbor_id)
        
        if processed_count != len(self.comfy_workflow):
            raise Exception("Cycle detected in ComfyUI graph or graph is disconnected.")
            
        # Add the streams that feed into the final output nodes as graph outputs
        node_configs = {str(n.get_id()): n for n in self.graph_config.get_nodes()}
        for output_node_id in execute_outputs:
            if output_node_id in node_configs:
                node_cfg = node_configs[output_node_id]
                for stream_cfg in node_cfg.get_input_streams():
                    self.graph_config.add_output_stream(stream_cfg)
        # Use generator mode so graph outputs are exposed for polling
        self.graph_config.set_mode("Generator")

        return self.graph_config
