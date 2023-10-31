import json
import os
import sys
import threading
import time

from bmf.lib._bmf import engine
from .bmf_modules import bmf_modules
from .bmf_node import BmfNode, BmfEdge
from .bmf_stream import BmfStream
from .ff_filter import get_filter_para
from .graph_config import NodeConfig, GraphConfigEncoder, GraphConfig, StreamConfig, ModuleConfig, MetaConfig
from ..ffmpeg_engine.engine import FFmpegEngine
from ..python_sdk import Log, LogLevel, Timestamp

## @defgroup pyAPI API in Python

if sys.version_info.major == 2:
    from Queue import Queue
else:
    from queue import Queue


class BmfCallBackType:
    LATEST_TIMESTAMP = 0


class GraphMode:
    NORMAL = 'Normal'  # indicate normal mode
    SERVER = 'Server'  # indicate server mode
    GENERATOR = 'Generator'  # indicate generator mode
    SUBGRAPH = 'Subgraph'  # indicate subgraph
    PUSHDATA = 'Pushdata'  # indicate push data
    FFMPEG = 'ffmpeg'
    C_ENGINE = 'c_engine'


## @ingroup pyAPI
## @defgroup grphClass BmfGraph
###@{
# BMF graph class
###@}
class BmfGraph:
    global_node_id_ = 0
    global_added_id_ = 0
    server_input_name = "server_input"
    node_id_mutex_ = threading.Lock()
    logbuffer_ = None
    av_log_list_ = list()

    def __init__(self, option=None):
        if option is None:
            option = {}
        self.mode = GraphMode.NORMAL
        self.nodes_ = []
        self.option_ = option

        # ignore graph output stream
        self.no_output_stream_ = option.get('no_output_stream', True)

        # graph input and output streams
        self.input_streams_ = []
        self.output_streams_ = []

        # save pre_created streams in SERVER mode
        self.node_streams_ = []

        # engine graph
        self.exec_graph_ = None
        self.graph_config_ = None
        self.update_graph_ = None

        # engine pre_allocated modules
        self.pre_module = {}

        # save created modules for sync mode
        self.sync_mode_ = {}

        # callbacks set by user
        self.user_callbacks = {}
        self.cb_lock = threading.RLock()

        if BmfGraph.logbuffer_ is not None:
            BmfGraph.logbuffer_.close()

    ## @ingroup pyAPI
    ## @ingroup grphClass
    ###@{
    #  set new graph options before run
    #  @param option: the option patch for the graph
    def set_option(self, option=None):
        ###@}
        if option is None:
            return

        for key in option:
            self.option_[key] = option[key]

    ## @ingroup pyAPI
    ## @ingroup grphClass
    ###@{
    #  To get a globalized effect buffer (list) which include all the log coming from ffmpeg libraries
    #
    #  @param level: ffmpeg av log level by default "info" level. it's optional, and also can be set:
    #                "quiet","panic","fatal","error","warning","info","verbose","debug","trace"
    #  @return A list object in python
    #  @note Should be called BEFORE graph run since it will notice the ffmpeg modulethat the log buffer is needed,
    #  the buffer will be clean each time when this function called
    def get_av_log_buffer(self, level='info'):
        ###@}
        # ffmpeg log config
        from bmf.lib._bmf.sdk import LogBuffer
        BmfGraph.av_log_list_.clear()
        BmfGraph.logbuffer_ = LogBuffer(BmfGraph.av_log_list_, level)
        return BmfGraph.av_log_list_

    ## @ingroup pyAPI
    ## @ingroup grphClass
    ###@{
    #  get sync module by given alias
    #  @param  alias: a node tag given by user while building graph pipeline
    def get_module(self, alias):
        ###@}
        select_node = None

        # find node by alias
        for node in self.nodes_:
            if "alias" in node.get_option() and node.get_option(
            )["alias"] == alias:
                select_node = node
                break

        # alias not correct
        if select_node is None:
            raise Exception('cannot find node according to alias')

        # create sync module
        if alias not in self.sync_mode_:
            sync_mode = select_node.create_sync_module()
            self.sync_mode_[alias] = sync_mode

        return self.sync_mode_[alias]

    ## @ingroup pyAPI
    ## @ingroup grphClass
    ###@{
    #  To setup a user defined callback into the graph. The callback can be triggered in the module
    #  @param  cb_type: a value can be defined by user to distinguish which is the one to call in multiple callbacks
    #  @param  cb: the function for this callback
    def add_user_callback(self, cb_type, cb):
        ###@}
        self.cb_lock.acquire()
        cb_list = self.user_callbacks.get(cb_type, [])
        if len(cb_list) == 0:
            self.user_callbacks[cb_type] = cb_list
        if cb is not None:
            cb_list.append(cb)
        self.cb_lock.release()

    ## @ingroup pyAPI
    ## @ingroup grphClass
    ###@{
    #  Remove the user defined callback from the callback list
    #  @param  cb_type: a value can be defined by user to distinguish which is the one to call in multiple callbacks
    #  @param  cb: the function for this callback
    def remove_user_callback(self, cb_type, cb):
        ###@}
        self.cb_lock.acquire()
        cb_list = self.user_callbacks.get(cb_type, [])
        cb_list.remove(cb)
        self.cb_lock.release()

    def clear_user_callback(self, cb_type, cb):
        self.cb_lock.acquire()
        self.user_callbacks[cb_type] = []
        self.cb_lock.release()

    def callback_for_engine(self, cb_type, para):
        # TODO: here we locked all types, can optimize to lock one type
        self.cb_lock.acquire()
        res = bytes("", "ASCII")
        cb_list = self.user_callbacks.get(cb_type, [])
        for cb in cb_list:
            if cb is not None:
                res = cb(para)
                break
        self.cb_lock.release()
        return res

    @staticmethod
    def generate_node_id():
        BmfGraph.node_id_mutex_.acquire()
        result = BmfGraph.global_node_id_
        BmfGraph.global_node_id_ += 1
        BmfGraph.node_id_mutex_.release()
        return result

    @staticmethod
    def generate_add_id():
        BmfGraph.node_id_mutex_.acquire()
        result = BmfGraph.global_added_id_
        BmfGraph.global_added_id_ += 1
        BmfGraph.node_id_mutex_.release()
        return result

    def add_node(self, node):
        if node is not None:
            self.nodes_.append(node)

    def module(self,
               module_info,
               option=None,
               module_path="",
               entry="",
               input_manager='immediate',
               pre_module=None,
               scheduler=0,
               stream_alias=None):
        if option is None:
            option = {}
        if isinstance(module_info, str):
            return BmfNode(
                {
                    "name": module_info,
                    "type": "",
                    "path": module_path,
                    "entry": entry
                }, option, self, input_manager, pre_module,
                scheduler).stream(stream_alias=stream_alias)
        return BmfNode(module_info, option, self, input_manager, pre_module,
                       scheduler).stream(stream_alias=stream_alias)

    ## @ingroup moduleAPI
    ###@{
    #  A graph function to provide a build-in decoder BMF stream
    #  Include av demuxer and decoder
    #  @param decoder_para: the parameters for the decoder
    #  @return A BMF stream(s)
    def decode(self,
               decoder_para,
               type="",
               path="",
               entry="",
               stream_alias=None):
        ###@}
        module_info = {
            "name": bmf_modules['ff_decoder'],
            "type": type,
            "path": path,
            "entry": entry
        }
        return BmfNode(module_info, decoder_para, self,
                       'immediate').stream(stream_alias=stream_alias)

    def download(self,
                 download_para,
                 type="",
                 path="",
                 entry="",
                 stream_alias=None):
        module_info = {
            "name": 'download',
            "type": type,
            "path": path,
            "entry": entry
        }
        return BmfNode(module_info, download_para, self,
                       'immediate').stream(stream_alias=stream_alias)

    def py_module(self,
                  name,
                  option=None,
                  module_path="",
                  entry="",
                  input_manager='immediate',
                  pre_module=None,
                  scheduler=0,
                  stream_alias=None):
        if option is None:
            option = {}
        return self.module(
            {
                "name": name,
                "type": "python",
                "path": module_path,
                "entry": entry
            },
            option,
            input_manager=input_manager,
            pre_module=pre_module,
            scheduler=scheduler,
            stream_alias=stream_alias)

    def go_module(self,
                  name,
                  option=None,
                  module_path="",
                  entry="",
                  input_manager="immediate",
                  pre_module=None,
                  scheduler=0,
                  stream_alias=None):
        if option is None:
            option = {}
        return self.module(
            {
                "name": name,
                "type": "go",
                "path": module_path,
                "entry": entry
            },
            option,
            input_manager=input_manager,
            pre_module=pre_module,
            scheduler=scheduler,
            stream_alias=stream_alias)

    ## @ingroup pyAPI
    ## @ingroup grphClass
    ###@{
    #  Using the stream in the graph to build a c/c++ implemented module stream loaded by module library path
    #  and entry
    #  @param name: the module name
    #  @param option: the parameters for the module
    #  @param module_path: the path to load the module
    #  @param entry: the call entry of the module
    #  @param input_manager: select the input manager for this module, immediate by default
    #  @return Stream(s) of the module
    def c_module(self,
                 name,
                 option=None,
                 module_path="",
                 entry="",
                 input_manager="immediate",
                 pre_module=None,
                 scheduler=0,
                 stream_alias=None):
        ###@}
        if option is None:
            option = {}
        return self.module(
            {
                "name": name,
                "type": "c++",
                "path": module_path,
                "entry": entry
            },
            option,
            input_manager=input_manager,
            pre_module=pre_module,
            scheduler=scheduler,
            stream_alias=stream_alias)

    def anullsrc(self, *args, **kwargs):
        stream_alias = None
        type = ""
        path = ""
        entry = ""
        if 'stream_alias' in kwargs:
            stream_alias = kwargs['stream_alias']
            del kwargs['stream_alias']
        if 'type' in kwargs:
            type = kwargs['type']
            del kwargs['type']
        if 'path' in kwargs:
            path = kwargs['path']
            del kwargs['path']
        if 'entry' in kwargs:
            entry = kwargs['entry']
            del kwargs['entry']

        para = get_filter_para(*args, **kwargs)
        if para is not None and len(para) > 0:
            option = {'name': 'anullsrc', 'para': para}
        module_info = {
            "name": bmf_modules['ff_filter'],
            "type": type,
            "path": path,
            "entry": entry
        }
        # create node
        return BmfNode(module_info, option, self,
                       'immediate').stream(stream_alias=stream_alias)

    def input_stream(self, name):
        stream = BmfStream(name, self, name)
        self.input_streams_.append(stream)
        return stream

    def fill_packet(self, name, packet, block=False):
        if self.exec_graph_ is not None:
            # pq = Queue()
            # pq.put(packet)
            self.exec_graph_.add_input_stream_packet(name, packet, block)

    def fill_eos(self, name):
        if self.exec_graph_ is not None:
            self.exec_graph_.add_eos_packet(name)

    def poll_packet(self, name, block=False):
        if self.exec_graph_ is not None:
            return self.exec_graph_.poll_output_stream_packet(name, block)
        else:
            time.sleep(1)

    @staticmethod
    def get_node_output_stream_map(node):
        stream_map = {}
        for edge in node.get_outgoing_edges():
            stream_map[edge.get_upstream_stream().get_notify(
            )] = edge.get_upstream_stream()
        return stream_map

    @staticmethod
    def all_stream_has_notify(stream_map):
        for notify in stream_map.keys():
            if not isinstance(notify, str):
                return False
        return True

    @staticmethod
    def all_stream_has_index(stream_map):
        max_index = -1
        for notify in stream_map.keys():
            if not isinstance(notify, int):
                return False, 0
            else:
                max_index = max(max_index, notify)

        return True, max_index

    @staticmethod
    def generate_node_stream_config(stream_map, node):
        streams = []
        if len(stream_map) == 0:
            return streams

        # all streams has notify
        if BmfGraph.all_stream_has_notify(stream_map):
            for (_, stream) in stream_map.items():
                stream_config = StreamConfig()
                stream_config.set_identifier(stream.get_identifier())
                if stream.get_alias() is None:
                    stream_config.set_alias("")
                else:
                    stream_config.set_alias(stream.get_alias())
                streams.append(stream_config)
            return streams

        # all streams don't have notify, use stream index as notify
        ret, max_index = BmfGraph.all_stream_has_index(stream_map)
        if ret:
            for index in range(max_index + 1):
                stream_config = StreamConfig()
                if index in stream_map.keys():
                    if stream_map[index].get_alias() is None:
                        stream_config.set_alias("")
                    else:
                        stream_config.set_alias(stream_map[index].get_alias())
                    stream_config.set_identifier(
                        stream_map[index].get_identifier())
                    streams.append(stream_config)
                else:
                    # just generate an unique name and hold the position
                    stream_config.set_identifier(node.generate_stream_name())
                    stream_config.set_alias("")
                    streams.append(stream_config)
            return streams

        print('failed to generate node stream config for ', node.get_type(),
              node.get_id())
        return streams

    @staticmethod
    def generate_module_info_config(module_info_dict):
        module_info_config = ModuleConfig()

        # set module name
        if module_info_dict.get('name'):
            module_info_config.set_name(module_info_dict['name'])
        else:
            module_info_config.set_name('')

        # set module type
        if module_info_dict.get('type'):
            module_info_config.set_type(module_info_dict['type'])
        else:
            module_info_config.set_type('')

        # set module path
        if module_info_dict.get('path'):
            module_info_config.set_path(module_info_dict['path'])
        else:
            module_info_config.set_path('')

        # set module entry
        if module_info_dict.get('entry'):
            module_info_config.set_entry(module_info_dict['entry'])
        else:
            module_info_config.set_entry('')

        return module_info_config

    @staticmethod
    def generate_meta_info_config(pre_module, callback_dict):
        meta_info_config = MetaConfig()

        # set pre_module
        if pre_module is not None:
            meta_info_config.set_premodule_id(pre_module.uid())
        # set callback function
        for key, callback in callback_dict.items():
            callback_binding = "{}:{}".format(key, callback[0])
            meta_info_config.add_callback_binding(callback_binding)

        return meta_info_config

    @staticmethod
    def generate_node_config(node):
        input_stream_map = node.get_input_streams()
        output_stream_map = BmfGraph.get_node_output_stream_map(node)

        node_config = NodeConfig()

        # set node id
        node_config.set_id(node.get_id())

        # set option
        node_config.set_option(node.get_option())

        # set module info
        node_config.set_module_info(
            BmfGraph.generate_module_info_config(node.get_module_info()))

        # set meta info
        node_config.set_meta_info(
            BmfGraph.generate_meta_info_config(node.get_pre_module(),
                                               node.get_user_callback()))

        # set alias
        node_config.set_alias(node.get_option().get('alias', ''))

        # set scheduler index
        node_config.set_scheduler(node.get_scheduler())

        # set input manager
        node_config.set_input_manager(node.get_input_manager())

        # set input streams
        node_config.set_input_streams(
            BmfGraph.generate_node_stream_config(input_stream_map, node))

        # set output streams
        node_config.set_output_streams(
            BmfGraph.generate_node_stream_config(output_stream_map, node))

        return node_config

    def dump_graph(self, graph_config):
        dump = self.option_.get('dump_graph', 0)

        graph_str = json.dumps(obj=graph_config.__dict__,
                               ensure_ascii=False,
                               indent=4,
                               cls=GraphConfigEncoder)

        # print(graph_str)
        Log.log(LogLevel.DEBUG, graph_str)

        if dump == 1:
            if 'graph_name' in self.option_:
                file_name = 'original_' + self.option_['graph_name'] + '.json'
            else:
                file_name = 'original_graph.json'

            f = open(file_name, 'w')
            f.write(graph_str)
            f.close()

    def generate_graph_config(self):
        graph_config = GraphConfig()

        # set option
        graph_config.set_option(self.option_)

        # set input stream
        for stream in self.input_streams_:
            stream_config = StreamConfig()
            stream_config.set_identifier(stream.get_name())
            if stream.get_alias() is None:
                stream_config.set_alias("")
            else:
                stream_config.set_alias(stream.get_alias())
            graph_config.add_input_stream(stream_config)

        # set output stream
        for stream in self.output_streams_:
            stream_config = StreamConfig()
            stream_config.set_identifier(stream.get_name())
            if stream.get_alias() is None:
                stream_config.set_alias("")
            else:
                stream_config.set_alias(stream.get_alias())
            graph_config.add_output_stream(stream_config)

        # node config
        for node in self.nodes_:
            node_config = BmfGraph.generate_node_config(node)
            graph_config.add_node_config(node_config)

        # graph pre_allocated module
        graph_pre_module = {}
        for node in self.nodes_:
            if node.get_pre_module() is not None:
                graph_pre_module[node.get_id()] = node.get_pre_module()

        # set graph mode
        graph_config.set_mode(self.mode)

        return graph_config, graph_pre_module

    def parse_output_streams(self, streams):
        if streams is not None:
            if isinstance(streams, BmfStream):
                # create a edge connected with stream and graph output stream
                graph_output_stream = BmfStream(streams.get_name(), None, 0)
                edge = BmfEdge(streams, graph_output_stream)
                streams.get_node().add_outgoing_edge(edge)
                self.output_streams_.append(graph_output_stream)
            elif isinstance(streams, list):
                for stream in streams:
                    if stream is not None:
                        graph_output_stream = BmfStream(
                            stream.get_name(), None, 0)
                        edge = BmfEdge(stream, graph_output_stream)
                        stream.get_node().add_outgoing_edge(edge)
                        self.output_streams_.append(graph_output_stream)

    def get_graph_config(self):
        return self.graph_config_

    ## @ingroup pyAPI
    ## @ingroup grphClass
    ###@{
    #  To run a graph by a graph config file
    #  @param graph_config: the graph config file path
    #  @return The name list of output streams in this graph
    def run_by_config(self, graph_config):
        ###@}
        self.dump_graph(graph_config)

        graph_config_str = graph_config.dump()
        # print(self.callback_for_engine)
        self.exec_graph_ = engine.Graph(
            graph_config_str, False,
            graph_config.get_option().get('optimize_graph', True))
        self.exec_graph_.start()

        # if graph has no input stream, 'close' will wait all nodes finish
        # else, we need fill packets to input stream and close graph manually
        if len(self.input_streams_) == 0 and len(self.output_streams_) == 0:
            self.exec_graph_.close()
        elif len(self.output_streams_) > 0:
            # return output stream name which is used to poll packets
            output_streams_name = []
            for stream in self.output_streams_:
                output_streams_name.append(stream.get_name())
            return output_streams_name

    ## @ingroup pyAPI
    ## @ingroup grphClass
    ###@{
    #  To generate the graph config only, without running
    #  @param streams: the input stream list of the module
    #  @param is_sub_graph: bool value to indicate whether it's a sub graph, False by default
    #  @param mode: to set the graph mode, NORMAL by default, other option bmf_graph.GraphMode
    #  @param file_name: output file name with extension
    def generate_config_file(self,
                             streams=None,
                             is_sub_graph=False,
                             mode=GraphMode.NORMAL,
                             is_blocked=True,
                             file_name="original_graph.json"):
        ###@}
        self.mode = mode

        # in server mode, graph has output_stream
        if self.mode == GraphMode.SERVER:
            self.no_output_stream_ = False

        # init graph output streams, support multi outputs
        # ignore output stream for main graph as default
        if not self.no_output_stream_ or is_sub_graph:
            self.parse_output_streams(streams)

        # in server mode, we should create an input_stream for graph, and also add it to the first node
        if self.mode == GraphMode.SERVER:
            if len(self.input_streams_) == 0:
                stream = self.input_stream(self.server_input_name)
            else:
                stream = self.input_streams_[0]
            # self.nodes_[0].input_streams_[0] = stream
            self.nodes_[0].init_input_stream_and_edge(stream, 0)
            for node in self.nodes_:
                node.set_input_manager('server')

        # parse graph config
        self.graph_config_, self.pre_module = self.generate_graph_config()
        if file_name != "":
            # save config file
            f = open(file_name, 'w')
            f.write(self.graph_config_.dump())
            f.close()

    ## @ingroup pyAPI
    ###@{
    #  To run the graph until it's finished
    #  @param streams: the input stream list of the module
    #  @param is_sub_graph: bool value to indicate whether it's a sub graph, False by default
    #  @param mode: to set the graph mode, NORMAL by default, other option bmf_graph.GraphMode
    def run(self,
            streams=None,
            is_sub_graph=False,
            mode=GraphMode.NORMAL,
            is_blocked=True):
        ###@}
        file_name = ""
        if 'dump_graph' in self.option_ and self.option_['dump_graph'] == 1:
            file_name = "original_graph.json"

        self.generate_config_file(streams=streams,
                                  is_sub_graph=is_sub_graph,
                                  mode=mode,
                                  is_blocked=is_blocked,
                                  file_name=file_name)

        graph_config_str = self.graph_config_.dump()
        print(graph_config_str)
        # call engine
        self.exec_graph_ = engine.Graph(
            graph_config_str, False, self.option_.get('optimize_graph', True))
        self.exec_graph_.start()

        # if graph has no input stream, 'close' will wait all nodes finish
        # else, we need fill packets to input stream and close graph manually
        if len(self.input_streams_) == 0 and len(self.output_streams_) == 0:
            if is_blocked:
                self.exec_graph_.close()
            else:
                print("start to run without block")
        elif len(self.output_streams_) > 0:
            # return output stream name which is used to poll packets
            output_streams_name = []
            for stream in self.output_streams_:
                output_streams_name.append(stream.get_name())
            return output_streams_name

        return None

    ## @ingroup pyAPI
    ## @ingroup grphClass
    ###@{
    #  Run the graph without wait to close, user should call close() by themself
    def run_wo_block(self,
                     streams=None,
                     is_sub_graph=False,
                     mode=GraphMode.NORMAL):
        ###@}
        return self.run(streams, is_sub_graph, mode, False)

    def runFFmpegByConfig(self, config_path):
        start_time = time.time()
        self.graph_config_ = GraphConfig(config_path)
        ffmpeg_engine = FFmpegEngine()
        command = ""
        if (ffmpeg_engine.is_valid_for_ffmpeg(self.graph_config_)):
            # self.dump_graph(self.graph_config_)
            command = ffmpeg_engine.get_ffmpeg_command(self.graph_config_)
            command = command + " -y"
        # do graph optimization
        print("ffmpeg command: ", command)
        os.system(command)
        end_time = time.time()
        ffmpeg_time = (end_time - start_time)
        return ffmpeg_time

    def start(self, stream, is_sub_graph=False):
        self.output_streams_.append(stream)

        # create a edge connected with stream and graph output stream
        graph_output_stream = BmfStream(stream.get_name(), None, 0)
        edge = BmfEdge(stream, graph_output_stream)
        stream.get_node().add_outgoing_edge(edge)
        if stream is not None:
            self.mode = GraphMode.GENERATOR

        # parse graph config
        self.graph_config_, self.pre_module = self.generate_graph_config()

        # for sub-graph, don't start executing
        if is_sub_graph:
            return

        # create and run graph
        graph_config_str = self.graph_config_.dump()
        self.exec_graph_ = engine.Graph(graph_config_str, False, True)
        self.exec_graph_.start()

        while True:
            pkt = self.exec_graph_.poll_output_stream_packet(
                stream.get_name(), True)
            if pkt is not None and pkt.defined():
                if pkt.timestamp == Timestamp.EOF:
                    break
                yield pkt

        self.exec_graph_.close()

    ## @ingroup pyAPI
    ## @ingroup grphClass
    ###@{
    #  To generate the graph of dynamical remove node, the graph should be different from running main graph.
    #  @param option: json style of description of which node to be removed
    #                 exp. {'alias': 'decode1'}
    def dynamic_remove(self, option):
        ###@}
        alias_name = option.get('alias', '')
        if len(alias_name) == 0:
            Log.log(LogLevel.ERROR,
                    "the alias name is must needed for removing")
            return False

        self.graph_ = BmfGraph(option)
        remove_node = BmfNode(alias_name, option, self, 'immediate')
        self.graph_.add_node(self)
        self.graph_config_, pre_module = self.generate_graph_config()
        for node_config in self.graph_config_.nodes:
            node_config.set_action('remove')

    ## @ingroup pyAPI
    ## @ingroup grphClass
    ###@{
    #  To generate the graph of dynamical add node, the graph should be different from running main graph.
    #  @param module_stream: the stream(s) of the new node
    #  @param inputs: a json style description for the input to be connected with this new node
    #                exp. {'alias': 'layout', 'streams': 1}
    #                it means the input of this node will be "layout" alias node and have 1 stream linked
    #  @param outputs: a json style description for the output to be connected with this new node
    def dynamic_add(self, module_stream, inputs=None, outputs=None):
        ###@}
        nb_links = 0
        add_id = 0

        self.graph_config_, pre_module = self.generate_graph_config()
        if inputs is not None:
            if module_stream.get_node().get_graph().graph_config_ is None:
                module_stream.get_node().get_graph(
                ).graph_config_, tmp = module_stream.get_node().get_graph(
                ).generate_graph_config()
                Log.log(LogLevel.ERROR,
                        "generate graph config for none graph config")

        tail_config = None
        for node_config in module_stream.get_node().get_graph(
        ).graph_config_.nodes:
            node_config.set_action('add')
            tail_config = node_config

        out_link_module_alias = ''
        if outputs is not None:
            out_link_module_alias = outputs.get('alias', '')
            nb_links = outputs.get('streams', 0)
            if tail_config is None:
                Log.log(LogLevel.ERROR,
                        "the output node config can't be found")
                return False
            add_id = self.generate_add_id()
            for i in range(nb_links):
                stream_config = StreamConfig()
                out_link_name = out_link_module_alias + "." + str(
                    add_id) + "_" + str(i)
                stream_config.set_identifier(out_link_name)
                stream_config.set_alias(out_link_name)
                tail_config.add_output_stream(stream_config)

        if inputs is not None:
            in_link_module_alias = inputs.get('alias', '')
            nb_links = inputs.get('streams', 0)
            ncfg = None
            for node_config in module_stream.get_node().get_graph(
            ).graph_config_.nodes:
                if len(node_config.get_input_streams()) == 0:
                    ncfg = node_config
                    break
            if ncfg is None:
                Log.log(LogLevel.ERROR, "the input node config can't be found")
                return False
            add_id = self.generate_add_id()
            for i in range(nb_links):
                stream_config = StreamConfig()
                in_link_name = in_link_module_alias + "." + str(
                    add_id) + "_" + str(i)
                stream_config.set_identifier(in_link_name)
                stream_config.set_alias(in_link_name)
                ncfg.add_input_stream(stream_config)

        graph_config_str = self.graph_config_.dump()
        self.exec_graph_ = engine.Graph(
            graph_config_str, False, self.option_.get('optimize_graph', True))

    ## @ingroup pyAPI
    ## @ingroup grphClass
    ###@{
    #  To generate the graph of dynamical node option reset, the graph should be different from running main         graph.
    #  @param option: json style of description of the parameters to be reset of the node
    #              exp. {'alias': 'encode1',
    #                    'output_path': output_path,
    #                    'video_params': {
    #                          'codec': 'h264',
    #                          'width': 320,
    #                          'height': 240,
    #                          'crf': 23,
    #                          'preset': 'veryfast'
    #                      }
    #                   }
    def dynamic_reset(self, option):
        ###@}
        alias_name = option.get('alias', '')
        if len(alias_name) == 0:
            Log.log(LogLevel.ERROR, "the alias name is must needed for reset")
            return False

        self.graph_ = BmfGraph(option)
        reset_node = BmfNode("", option, self)
        self.graph_.add_node(self)
        self.graph_config_, pre_module = self.generate_graph_config()
        for node_config in self.graph_config_.nodes:
            node_config.set_action('reset')

    ## @ingroup pyAPI
    ## @ingroup grphClass
    ###@{
    #  Final action to do the dynamical add/remove/reset node for current running graph.
    #  @param update_graph: the graph generated by previous dynamic_add(), dynamic_remove() or dynamic_reset()
    def update(self, update_graph):
        ###@}
        if update_graph is None or update_graph.graph_config_ is None:
            Log.log(LogLevel.ERROR,
                    "the graph for update is not created properly")
            return False

        graph_config_str = update_graph.graph_config_.dump()

        self.exec_graph_.update(graph_config_str, False)

    def status(self):
        if self.exec_graph_ is not None:
            return self.exec_graph_.status()
        return None

    ## @ingroup pyAPI
    ## @ingroup grphClass
    ###@{
    # To close the graph by block wait until all the tasks are finished.
    def close(self):
        ###@}
        if self.exec_graph_ is not None:
            self.exec_graph_.close()

    ## @ingroup pyAPI
    ## @ingroup grphClass
    ###@{
    #  Force close the running graph even if the whole pipeline in the graph is not finished
    def force_close(self):
        ###@}
        if self.exec_graph_ is not None:
            self.exec_graph_.force_close()

    def generateConfig(self, file_name):
        self.graph_config_, graph_pre_module = self.generate_graph_config()
        print(self.graph_config_)
        self.dump_graph(self.graph_config_)
        graph_str = json.dumps(obj=self.graph_config_.__dict__,
                               ensure_ascii=False,
                               indent=4,
                               cls=GraphConfigEncoder)
        f = open(file_name, 'w')
        f.write(graph_str)
        f.close()

    # there will be some error when bmf_graph Deconstruction, it may use thread_queue thread to handle Deconstruction
    # which will cause hang
    # def __del__(self):
    #     if self.exec_graph_ is not None:
    #         self.exec_graph_.close()
