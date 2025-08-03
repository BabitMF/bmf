
import bmf
from bmf import Module, Task, Packet, Timestamp, ProcessResult
import sys
import os
import torch
import dill 
from bmf.builder.graph_config import GraphConfig, NodeConfig, StreamConfig, ModuleConfig

# Add ComfyUI to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../ComfyUI')))
import nodes

class ComfyNodeRunner(Module):
    def __init__(self, node_id=None, option=None):
        super().__init__(node_id, option)
        self.option = option
        
        self.class_type = option['class_type']
        self.comfy_node_class = nodes.NODE_CLASS_MAPPINGS[self.class_type]
        self.comfy_node_instance = self.comfy_node_class()
        
        self.widget_inputs = {}
        self.link_inputs_info = {} 

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

    def process(self, task):
        kwargs = self.widget_inputs.copy()
        
        # Unpack inputs from BMF packets
        for i in range(len(task.get_inputs())):
            input_queue = task.get_inputs()[i]
            if not input_queue.empty():
                pkt = input_queue.get()
                if pkt.timestamp != Timestamp.EOF:
                    input_name = self.link_inputs_info[i]
                    data_bytes = pkt.get(bytes)
                    unwrapped_data = dill.loads(data_bytes)
                    kwargs[input_name] = unwrapped_data

        function_name = getattr(self.comfy_node_instance, 'FUNCTION', 'execute')
        execute_func = getattr(self.comfy_node_instance, function_name)
        
        with torch.inference_mode():
             results = execute_func(**kwargs)

        # Wrap outputs in BMF packets and send
        if results:
            for i, res_item in enumerate(results):
                if i in task.get_outputs():
                    data_bytes = dill.dumps(res_item)
                    out_pkt = Packet(data_bytes)
                    out_pkt.timestamp = task.timestamp
                    task.get_outputs()[i].put(out_pkt)
                    
        for i in task.get_outputs():
            task.get_outputs()[i].put(Packet.generate_eof_packet())

        return ProcessResult.OK

class BmfWorkflowConverter:
    def __init__(self, comfy_workflow, server_instance):
        self.comfy_workflow = comfy_workflow
        self.server = server_instance
        self.graph_config = GraphConfig()
        self.graph_config.set_option({"dump_graph": 1, "graph_name": "comfy_to_bmf"})

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
                "comfy_node_id": node_id
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
                for input_name in input_order:
                    if input_name in links:
                        link_val = links[input_name]
                        from_node_id, from_port = str(link_val[0]), link_val[1]
                        stream_identifier = node_outputs.get((from_node_id, from_port))
                        if stream_identifier:
                            stream_conf = StreamConfig({"identifier": stream_identifier, "stream_alias": ""})
                            node_config.add_input_stream(stream_conf)

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

        return self.graph_config
