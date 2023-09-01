import json
import bmf
from bmf import Task, Packet
from bmf.lib._bmf import engine

## @ingroup pyAPI
## @defgroup syncMd SyncModule
###@{
# BMF sync module class
###@}


class SyncModule(object):

    def __init__(self, mod, input_streams, output_streams):
        self.mod = mod
        self.input_streams = input_streams
        self.output_streams = output_streams

    def get_input_streams(self):
        return self.input_streams

    def get_output_streams(self):
        return self.output_streams

    def init(self):
        self.mod.init()

    def process(self, task):
        self.mod.process(task)

    def process_pkts(self, pkts_dict):
        # create task
        task = Task(0, self.get_input_streams(), self.get_output_streams())

        # fill in task inputs
        if pkts_dict is not None:
            for key, pkts in pkts_dict.items():
                if key not in self.get_input_streams():
                    raise Exception("pkt key not exists")
                for packet in pkts:
                    task.get_inputs()[key].put(packet)

        self.mod.process(task)

        # get task outputs
        result_dict = {}
        for (key, q) in task.get_outputs().items():
            result_dict[key] = []
            while not q.empty():
                result_dict[key].append(q.get())

        return result_dict, task.timestamp

    def send_eof(self):
        # create task
        task = Task(0, self.get_input_streams(), self.get_output_streams())

        # send eof to task
        for key in self.get_input_streams():
            task.get_inputs()[key].put(Packet.generate_eof_packet())

        self.mod.process(task)

    def close(self):
        self.mod.close()


## @ingroup pyAPI
## @ingroup syncMd
###@{
#  Create SyncModule by name, option, input_stream_id_list and output_stream_id_list
#  @param name: the name for the module
#  @param name: the option for the module
#  @param name: the input stream id list for the module
#  @param name: the output stream id list for the module
#  @return a syncModule object which contains a C++ module inside
def sync_module(name, option, input_streams, output_streams):
    ###@}
    # convert module option for filter module
    if name == "c_ffmpeg_filter":
        # construct node config
        node_config = {}
        node_config["option"] = option
        node_config["input_streams"] = []
        for index in input_streams:
            input_stream = {"identifier": name + str(index)}
            node_config["input_streams"].append(input_stream)
        node_config["output_streams"] = []
        for index in output_streams:
            output_stream = {"identifier": name + str(index)}
            node_config["output_streams"].append(output_stream)

        # convert filter option
        option_str = engine.convert_filter_para(json.dumps(node_config))
        option = json.loads(option_str)

    # Directly create a C++ module by module name and option
    mod = bmf.create_module(name, option)

    return SyncModule(mod, input_streams, output_streams)


## @ingroup pyAPI
## @ingroup syncMd
###@{
#  Directly do module processing
#  @param module: corresponding syncModule object
#  @param pkts_dict: a dict which contains all input data packet
#  @return a dict which contains result data packet
#  @return task timestamp
def process(module, pkts_dict):
    ###@}
    # create task
    task = Task(0, module.get_input_streams(), module.get_output_streams())

    # fill in task inputs
    if pkts_dict is not None:
        for key, pkts in pkts_dict.items():
            if key not in module.get_input_streams():
                raise Exception("pkt key not exists")
            for packet in pkts:
                task.get_inputs()[key].put(packet)

    # process task
    module.process(task)

    # get task outputs
    result_dict = {}
    for (key, q) in task.get_outputs().items():
        result_dict[key] = []
        while not q.empty():
            result_dict[key].append(q.get())

    return result_dict, task.timestamp


## @ingroup pyAPI
## @ingroup syncMd
###@{
#  Module process a task with eof packet
#  @param module: corresponding syncModule object
def send_eof(module):
    ###@}
    # create task
    task = Task(0, module.get_input_streams(), module.get_output_streams())

    # send eof to task
    for key in module.get_input_streams():
        task.get_inputs()[key].put(Packet.generate_eof_packet())

    # process eof task
    module.process(task)

    # get task outputs
    result_dict = {}
    for (key, q) in task.get_outputs().items():
        result_dict[key] = []
        while not q.empty():
            result_dict[key].append(q.get())

    return result_dict, task.timestamp
