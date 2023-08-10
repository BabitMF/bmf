import json
from json import JSONEncoder
import string
# from .utils import Log, LogLevel
from ..python_sdk import Log, LogLevel
import sys


class GraphConfigEncoder(JSONEncoder):

    def default(self, o):
        return o.__dict__


class GraphConfig:
    """
    {
        "option": {},
        "input_streams":[
            "in"
        ],
        "output_streams":[
            "out"
        ],
        "nodes":[
            {
                "id":1,
                "scheduler":0
                "module_info":{.....}
                "meta_info":{.....}
                "option":{.....}
                "input_manager":"immediate"
                "input_streams": [
                    {.....}
                    {.....}
                ],
                "output_streams": [
                    {.....}
                    {.....}
                ]
            }
        ],
        "mode": 'normal'
    }
    """

    def __init__(self, config_file=None):
        self.input_streams = []
        self.output_streams = []
        self.nodes = []
        self.option = {}
        self.mode = 'normal'

        if config_file is not None:
            # load config file
            with open(config_file, mode='r') as f:
                config_dict = json.loads(f.read())
                self.parse(config_dict)

    def parse(self, content):
        # turn unicode to str
        config_dict = self.unicode_convert(content)

        # check if the config file is correct
        # if not self.validate_graph_config_file(config_dict):
        #     Log.log(LogLevel.ERROR, "Config file is not correct")
        #     return

        if 'input_streams' in config_dict.keys():
            for input_stream_dict in config_dict['input_streams']:
                self.input_streams.append(StreamConfig(input_stream_dict))

        if 'output_streams' in config_dict.keys():
            for output_stream_conifg in config_dict['output_streams']:
                self.output_streams.append(StreamConfig(output_stream_conifg))

        for node_config in config_dict['nodes']:
            self.nodes.append(NodeConfig(node_config))

        if 'option' in config_dict.keys():
            self.option = config_dict['option']

    def unicode_convert(self, input):
        if isinstance(input, dict):
            return {
                self.unicode_convert(key): self.unicode_convert(value)
                for key, value in input.items()
            }
        elif isinstance(input, list):
            return [self.unicode_convert(element) for element in input]
        elif sys.version_info.major == 2 and isinstance(input, unicode):
            return input.encode('utf-8')
        else:
            return input

    def get_nodes(self):
        return self.nodes

    def get_option(self):
        return self.option

    def get_input_streams(self):
        return self.input_streams

    def get_output_streams(self):
        return self.output_streams

    def get_input_stream_names(self):
        input_stream_names = []
        for input_stream in self.get_input_streams():
            input_stream_names.append(input_stream.get_identifier())
        return input_stream_names

    def get_output_stream_names(self):
        output_stream_names = []
        for output_stream in self.get_output_streams():
            output_stream_names.append(output_stream.get_identifier())
        return output_stream_names

    def add_input_stream(self, stream):
        if stream is not None:
            self.input_streams.append(stream)

    def add_output_stream(self, stream):
        if stream is not None:
            self.output_streams.append(stream)

    def set_option(self, option):
        if option is not None:
            self.option = option

    def get_mode(self):
        return self.mode

    def set_mode(self, mode):
        self.mode = mode

    def add_node_config(self, node_config):
        if node_config is not None:
            self.nodes.append(node_config)

    def dump(self):
        return json.dumps(obj=self.__dict__,
                          ensure_ascii=False,
                          indent=4,
                          cls=GraphConfigEncoder)


class NodeConfig:
    """
    {
        "id":1,
        "scheduler":0
        "module_info":{.....}
        "meta_info":{.....}
        "option":{.....}
        "input_manager":"immediate"
        "input_streams": [
            {.....}
            {.....}
        ],
        "output_streams": [
            {.....}
            {.....}
        ]
    }
    """

    def __init__(self, node_config_dict=None):
        self.module_info = {}
        self.meta_info = {}
        self.option = {}
        self.input_streams = []
        self.output_streams = []
        self.input_manager = 'default'
        self.scheduler = 0
        self.alias = ''

        if node_config_dict is not None:
            if 'id' in node_config_dict.keys():
                self.id = node_config_dict['id']

            if 'scheduler' in node_config_dict.keys():
                self.scheduler = node_config_dict['scheduler']

            if 'module_info' in node_config_dict.keys():
                self.module_info = ModuleConfig(
                    node_config_dict['module_info'])

            if 'meta_info' in node_config_dict.keys():
                self.meta_info = MetaConfig(node_config_dict['meta_info'])

            if 'option' in node_config_dict.keys():
                self.option = node_config_dict['option']

            if 'input_streams' in node_config_dict.keys():
                for input_stream_dict in node_config_dict['input_streams']:
                    self.input_streams.append(StreamConfig(input_stream_dict))

            if 'input_manager' in node_config_dict.keys():
                self.input_manager = node_config_dict['input_manager']

            if 'output_streams' in node_config_dict.keys():
                for output_stream_dict in node_config_dict['output_streams']:
                    self.output_streams.append(
                        StreamConfig(output_stream_dict))

    def add_input_stream(self, stream):
        self.input_streams.append(stream)

    def add_output_stream(self, stream):
        self.output_streams.append(stream)

    def get_input_streams(self):
        return self.input_streams

    def get_output_streams(self):
        return self.output_streams

    def set_input_streams(self, streams):
        self.input_streams = streams

    def get_input_manager(self):
        return self.input_manager

    def set_input_manager(self, manager):
        self.input_manager = manager

    def set_output_streams(self, streams):
        self.output_streams = streams

    def get_input_stream_names(self):
        input_stream_names = []
        for input_stream in self.get_input_streams():
            input_stream_names.append(input_stream.get_identifier())
        return input_stream_names

    def get_output_stream_names(self):
        output_stream_names = []
        for output_stream in self.get_output_streams():
            output_stream_names.append(output_stream.get_identifier())
        return output_stream_names

    def add_option(self, key, value):
        self.option[key] = value

    def get_option(self):
        return self.option

    def set_option(self, option):
        self.option = option

    def set_module_info(self, module_info):
        self.module_info = module_info

    def get_module_info(self):
        return self.module_info

    def set_meta_info(self, meta_info):
        self.meta_info = meta_info

    def get_meta_info(self):
        return self.meta_info

    def set_id(self, idx):
        self.id = idx

    def get_id(self):
        return self.id

    def set_alias(self, alias):
        self.alias = alias

    def set_action(self, act):
        self.action = act

    def set_scheduler(self, sched):
        self.scheduler = sched

    def get_scheduler(self):
        return self.scheduler

    def dump(self):
        return json.dumps(obj=self.__dict__,
                          ensure_ascii=False,
                          indent=4,
                          cls=GraphConfigEncoder)


class StreamConfig:
    """
        {
            "identifier": "video:ffmpeg_decoder_0_0",
            "stream_alias": "v0_video"
        }
    """

    def __init__(self, stream_config_dict=None):
        self.identifier = None
        self.stream_alias = None

        if stream_config_dict is not None:
            if 'identifier' in stream_config_dict.keys():
                self.identifier = stream_config_dict['identifier']

            if 'stream_alias' in stream_config_dict.keys():
                self.stream_alias = stream_config_dict['stream_alias']

    def set_identifier(self, identifier):
        self.identifier = identifier

    def get_identifier(self):
        return self.identifier

    def set_alias(self, stream_alias):
        self.stream_alias = stream_alias

    def get_alias(self):
        return self.stream_alias

    def dump(self):
        return json.dumps(obj=self.__dict__,
                          ensure_ascii=False,
                          indent=4,
                          cls=GraphConfigEncoder)


class ModuleConfig:
    """
        {
            "name": "c_ffmpeg_encoder",
            "type": "python",
            "path": "",
            "entry": ""
        }
    """

    def __init__(self, module_config_dict=None):
        self.name = None
        self.type = None
        self.path = None
        self.entry = None

        if module_config_dict is not None:
            if 'name' in module_config_dict.keys():
                self.name = module_config_dict['name']

            if 'type' in module_config_dict.keys():
                self.type = module_config_dict['type']

            if 'path' in module_config_dict.keys():
                self.path = module_config_dict['path']

            if 'entry' in module_config_dict.keys():
                self.entry = module_config_dict['entry']

    def set_name(self, name):
        self.name = name

    def get_name(self):
        return self.name

    def set_type(self, type):
        self.type = type

    def get_type(self):
        return self.type

    def set_path(self, path):
        self.path = path

    def get_path(self):
        return self.path

    def set_entry(self, entry):
        self.entry = entry

    def get_entry(self):
        return self.entry

    def dump(self):
        return json.dumps(obj=self.__dict__,
                          ensure_ascii=False,
                          indent=4,
                          cls=GraphConfigEncoder)


class MetaConfig:
    """
        {
            "premodule_id": -1,
            "callback_binding": [
                "100:1",
                "1:2",
                "999:3"
            ]
        }
    """

    def __init__(self, meta_config_dict=None):
        self.premodule_id = -1
        self.callback_binding = []

        if meta_config_dict is not None:
            if 'premodule_id' in meta_config_dict.keys():
                self.premodule_id = meta_config_dict['premodule_id']

            if 'callback_binding' in meta_config_dict.keys():
                self.callback_binding = meta_config_dict['callback_binding']

    def set_premodule_id(self, premodule_id):
        self.premodule_id = premodule_id

    def get_premodule_id(self):
        return self.premodule_id

    def add_callback_binding(self, callback_binding):
        self.callback_binding.append(callback_binding)

    def get_callback_binding(self):
        return self.callback_binding

    def dump(self):
        return json.dumps(obj=self.__dict__,
                          ensure_ascii=False,
                          indent=4,
                          cls=GraphConfigEncoder)
