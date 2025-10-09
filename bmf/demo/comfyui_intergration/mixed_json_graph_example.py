import sys
import os
import json
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
from demo.comfyui_intergration.bridge import BmfWorkflowConverter
from bmf.builder.graph_config import NodeConfig, StreamConfig, ModuleConfig
from bmf import Log, LogLevel, Timestamp, VideoFrame, bmf_convert, MediaDesc, MediaType

# Set BMF log level
Log.set_log_level(LogLevel.INFO)


class MockServer:
    def __init__(self):
        self.client_id = None
        self.last_prompt_id = None

    def send_sync(self, event, data, sid=None):
        Log.log(LogLevel.DEBUG, f"MockServer send_sync: event={event}")
        pass

def modify_graph_config(graph_config, comfy_workflow):
    """
    Inserts BMF native nodes into the graph config generated from a ComfyUI workflow.
    """
    nodes = {n.get_id(): n for n in graph_config.get_nodes()}

    # Node IDs from prompt.json: 8=VAEDecode, 9=SaveImage
    DECODE_NODE_ID = 8
    SAVE_NODE_ID = 9

    if DECODE_NODE_ID not in nodes or SAVE_NODE_ID not in nodes:
        raise ValueError("Could not find VAEDecode (8) or SaveImage (9) nodes in the graph.")

    node8_config = nodes[DECODE_NODE_ID]
    node9_config = nodes[SAVE_NODE_ID]

    # Find the stream from VAEDecode to SaveImage
    if not node9_config.get_input_streams():
        raise ValueError("SaveImage node has no input streams.")
    stream_8_to_9 = node9_config.get_input_streams()[0]

    # Find a new unique ID for our nodes
    max_id = max(int(nid) for nid in comfy_workflow.keys())
    tensor_to_vf_id = max_id + 1
    jpg_encode_id = max_id + 2

    Log.log(LogLevel.INFO, "Inserting BMF nodes between ComfyUI nodes 8 and 9")

    # 1. Create TensorToVideoFrame Python node (branching from VAEDecode without altering SaveImage)
    stream_t2vf_out = StreamConfig({"identifier": f"stream_t2vf_{tensor_to_vf_id}_0", "stream_alias": ""})
    t2vf_node = NodeConfig()
    t2vf_node.set_id(tensor_to_vf_id)
    t2vf_node.set_module_info(ModuleConfig({
        "name": "TensorToVideoFrame",
        "type": "python",
        "path": integration_dir,
        "entry": f"{os.path.splitext(os.path.basename(__file__))[0]}:TensorToVideoFrame"
    }))
    # Do NOT mutate SaveImage's input stream object; create a new reference to the same identifier
    t2vf_input = StreamConfig({
        "identifier": stream_8_to_9.get_identifier(),
        "stream_alias": stream_8_to_9.get_alias() if stream_8_to_9.get_alias() is not None else ""
    })
    t2vf_node.add_input_stream(t2vf_input)
    t2vf_node.add_output_stream(stream_t2vf_out)
    graph_config.add_node_config(t2vf_node)

    # 2. Add a parallel BMF Encode node to write a JPG from the VideoFrame branch
    encode_node = NodeConfig()
    encode_node.set_id(jpg_encode_id)
    encode_node.set_module_info(ModuleConfig({
        "name": "c_ffmpeg_encoder",
        "type": "c++",
        "path": "",
        "entry": ""
    }))
    # Feed encoder with VideoFrame from TensorToVideoFrame
    encode_node.add_input_stream(stream_t2vf_out)
    # Configure scheduling similar to bmf.encode default
    encode_node.set_scheduler(1)
    encode_node.set_input_manager('immediate')
    # Configure encoder to output JPG
    encode_node.set_option({
        "output_path": os.path.join(integration_dir, "json_output.jpg"),
        "format": "mjpeg",
        "video_params": {
            "codec": "jpg"
        }
    })
    graph_config.add_node_config(encode_node)
    # Keep SaveImage input untouched to preserve PNG output

    # 5. Normalize ComfyNodeRunner nodes to import via local bridge module
    for n in graph_config.get_nodes():
        mi = n.get_module_info()
        try:
            if mi.get_name() == "ComfyNodeRunner":
                # Force import using local bridge module to avoid relying on 'demo.' package root
                mi.set_entry("bridge:ComfyNodeRunner")
                mi.set_path(integration_dir)
        except Exception:
            pass

    Log.log(LogLevel.INFO, "Graph modification complete.")
    return graph_config

def main():
    # Load ComfyUI workflow files
    base_path = os.path.dirname(__file__)
    with open(os.path.join(base_path, 'example_input/prompt_t2i.json'), 'r') as f:
        prompt = json.load(f)
    with open(os.path.join(base_path, 'example_input/extra_data_t2i.json'), 'r') as f:
        extra_data = json.load(f)
    with open(os.path.join(base_path, 'example_input/execute_outputs_t2i.json'), 'r') as f:
        execute_outputs = json.load(f)
    
    Log.log(LogLevel.INFO, "Loaded ComfyUI workflow.")

    # Convert ComfyUI workflow to a BMF GraphConfig
    server_instance = MockServer()
    converter = BmfWorkflowConverter(prompt, server_instance, extra_data)
    graph_config = converter.convert(execute_outputs)

    # Modify the GraphConfig to insert native BMF nodes
    modified_graph_config = modify_graph_config(graph_config, prompt)
    
    # Optional: Dump the modified graph to a file for inspection
    with open("comfy_to_bmf_mixed_t2i.json", "w") as f:
        f.write(modified_graph_config.dump())
    Log.log(LogLevel.INFO, "Dumped modified graph to comfy_to_bmf_mixed_t2i.json")

    # Run the modified graph
    Log.log(LogLevel.INFO, "Running modified BMF graph...")
    bmf_graph = bmf.graph({"dump_graph": 1, "graph_name": "comfy_mixed_t2i"})
    bmf_graph.run_by_config(modified_graph_config)
    Log.log(LogLevel.INFO, "Graph execution finished.")

if __name__ == '__main__':
    main()
