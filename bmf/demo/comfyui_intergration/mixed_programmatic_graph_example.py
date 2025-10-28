import sys
import os
from utils import TensorToVideoFrame

# Ensure paths are set up correctly to find bmf and comfyui_integration modules
integration_dir = os.path.dirname(os.path.abspath(__file__))
bmf_output_dir = os.path.abspath(os.path.join(integration_dir, '..', '..'))
comfyui_path = os.path.join(integration_dir, 'ComfyUI')
# Ensure output-built bmf package takes precedence over source tree
for p in (bmf_output_dir, integration_dir, comfyui_path):
    if p in sys.path:
        sys.path.remove(p)
sys.path.insert(0, bmf_output_dir)
sys.path.insert(1, integration_dir)
sys.path.insert(2, comfyui_path)

# Ensure this script's module name is importable by entry="<module>:<Class>"
_module_name = os.path.splitext(os.path.basename(__file__))[0]
sys.modules.setdefault(_module_name, sys.modules[__name__])

import bmf
import bmf.hmp as mp
from bmf import Log, LogLevel, Timestamp, VideoFrame
from bmf.builder.graph_config import GraphConfig, NodeConfig, StreamConfig, ModuleConfig

# ComfyUI node registry
import nodes

# Set BMF log level
Log.set_log_level(LogLevel.INFO)




def _get_input_order(class_type: str):
    try:
        node_class = nodes.NODE_CLASS_MAPPINGS[class_type]
        input_types = node_class.INPUT_TYPES()
        required = list(input_types.get('required', {}).keys())
        optional = list(input_types.get('optional', {}).keys())
        return required + optional
    except Exception:
        return []


def _get_return_count(class_type: str) -> int:
    try:
        rt = nodes.NODE_CLASS_MAPPINGS[class_type].RETURN_TYPES
        return len(rt) if isinstance(rt, (tuple, list)) else 0
    except Exception:
        return 0


def build_programmatic_graph(image_name: str = "example.png",
                             ckpt_name: str = "v1-5-pruned-emaonly-fp16.safetensors",
                             output_path: str = None) -> GraphConfig:
    if output_path is None:
        output_path = os.path.join(integration_dir, "programmatic_output.jpg")

    graph = GraphConfig()
    graph.set_option({"dump_graph": 1, "graph_name": "comfy_programmatic_graph"})

    # Track stream identifiers per (node_id, port)
    stream_map = {}

    # Helper to add output streams for a Comfy node based on RETURN_TYPES
    def add_comfy_outputs(node_cfg: NodeConfig, class_type: str, node_id_str: str):
        out_count = _get_return_count(class_type)
        for i in range(out_count):
            sid = f"stream_{node_id_str}_{i}"
            sc = StreamConfig({"identifier": sid, "stream_alias": ""})
            node_cfg.add_output_stream(sc)
            stream_map[(node_id_str, i)] = sid

    # 1) CheckpointLoaderSimple (id "14")
    n_ckpt = NodeConfig()
    n_ckpt.set_id(14)
    n_ckpt.set_module_info(ModuleConfig({
        "name": "ComfyNodeRunner",
        "type": "python",
        "path": integration_dir,
        "entry": "bridge:ComfyNodeRunner"
    }))
    n_ckpt.set_option({
        "class_type": "CheckpointLoaderSimple",
        "inputs": {"ckpt_name": ckpt_name},
        "comfy_node_id": "14"
    })
    add_comfy_outputs(n_ckpt, "CheckpointLoaderSimple", "14")
    graph.add_node_config(n_ckpt)

    # 2) LoadImage (id "10")
    n_load = NodeConfig()
    n_load.set_id(10)
    n_load.set_module_info(ModuleConfig({
        "name": "ComfyNodeRunner",
        "type": "python",
        "path": integration_dir,
        "entry": "bridge:ComfyNodeRunner"
    }))
    n_load.set_option({
        "class_type": "LoadImage",
        "inputs": {"image": image_name},
        "comfy_node_id": "10"
    })
    add_comfy_outputs(n_load, "LoadImage", "10")
    graph.add_node_config(n_load)

    # 3) VAEEncode (id "12")
    n_encode = NodeConfig()
    n_encode.set_id(12)
    n_encode.set_module_info(ModuleConfig({
        "name": "ComfyNodeRunner",
        "type": "python",
        "path": integration_dir,
        "entry": "bridge:ComfyNodeRunner"
    }))
    # Link inputs: pixels <- LoadImage:0, vae <- CheckpointLoaderSimple:2
    n_encode_inputs = {"pixels": ["10", 0], "vae": ["14", 2]}
    n_encode.set_option({
        "class_type": "VAEEncode",
        "inputs": n_encode_inputs,
        "comfy_node_id": "12"
    })
    # Add input streams in canonical order
    order_encode = _get_input_order("VAEEncode")
    for name in order_encode:
        if name in n_encode_inputs and isinstance(n_encode_inputs[name], list):
            from_id, from_port = n_encode_inputs[name][0], n_encode_inputs[name][1]
            sid = stream_map[(from_id, from_port)]
            n_encode.add_input_stream(StreamConfig({"identifier": sid, "stream_alias": ""}))
    # Multi-input node: require framesync behavior so both inputs are available
    n_encode.set_input_manager('framesync')
    add_comfy_outputs(n_encode, "VAEEncode", "12")
    graph.add_node_config(n_encode)

    # 4) VAEDecode (id "8")
    n_decode = NodeConfig()
    n_decode.set_id(8)
    n_decode.set_module_info(ModuleConfig({
        "name": "ComfyNodeRunner",
        "type": "python",
        "path": integration_dir,
        "entry": "bridge:ComfyNodeRunner"
    }))
    n_decode_inputs = {"samples": ["12", 0], "vae": ["14", 2]}
    n_decode.set_option({
        "class_type": "VAEDecode",
        "inputs": n_decode_inputs,
        "comfy_node_id": "8"
    })
    order_decode = _get_input_order("VAEDecode")
    for name in order_decode:
        if name in n_decode_inputs and isinstance(n_decode_inputs[name], list):
            from_id, from_port = n_decode_inputs[name][0], n_decode_inputs[name][1]
            sid = stream_map[(from_id, from_port)]
            n_decode.add_input_stream(StreamConfig({"identifier": sid, "stream_alias": ""}))
    # Multi-input node: require framesync
    n_decode.set_input_manager('framesync')
    add_comfy_outputs(n_decode, "VAEDecode", "8")
    graph.add_node_config(n_decode)

    # 5) TensorToVideoFrame (id "101") – convert Comfy IMAGE tensor to BMF VideoFrame
    n_t2vf = NodeConfig()
    n_t2vf.set_id(101)
    n_t2vf.set_module_info(ModuleConfig({
        "name": "TensorToVideoFrame",
        "type": "python",
        "path": integration_dir,
        "entry": f"{os.path.splitext(os.path.basename(__file__))[0]}:TensorToVideoFrame"
    }))
    # Input from VAEDecode:0 (IMAGE)
    sid_decode_img = stream_map[("8", 0)]
    n_t2vf.add_input_stream(StreamConfig({"identifier": sid_decode_img, "stream_alias": ""}))
    # Single output
    sid_t2vf_out = "stream_t2vf_0"
    n_t2vf.add_output_stream(StreamConfig({"identifier": sid_t2vf_out, "stream_alias": ""}))
    graph.add_node_config(n_t2vf)

    # 6) c_ffmpeg_encoder (id "102") – save as JPG
    n_enc = NodeConfig()
    n_enc.set_id(102)
    n_enc.set_module_info(ModuleConfig({
        "name": "c_ffmpeg_encoder",
        "type": "c++",
        "path": "",
        "entry": ""
    }))
    n_enc.add_input_stream(StreamConfig({"identifier": sid_t2vf_out, "stream_alias": ""}))
    n_enc.set_scheduler(1)
    n_enc.set_input_manager('immediate')
    n_enc.set_option({
        "output_path": output_path,
        "format": "mjpeg",
        "video_params": {"codec": "jpg"}
    })
    graph.add_node_config(n_enc)

    return graph


def main():
    Log.log(LogLevel.INFO, "Building programmatic Comfy -> BMF graph (no JSON)...")
    graph_cfg = build_programmatic_graph()
    Log.log(LogLevel.INFO, "Running graph...")
    g = bmf.graph({"dump_graph": 1, "graph_name": "comfy_programmatic_graph"})
    g.run_by_config(graph_cfg)
    Log.log(LogLevel.INFO, "Done. Wrote programmatic_output.jpg if inputs were valid.")


if __name__ == '__main__':
    main()


