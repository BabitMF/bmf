import threading
from .bmf_stream import BmfStream
import json
from .bmf_sync import SyncModule


# edge is created by downstream node and add to outgoing edge list
# of upstream node
class BmfEdge:

    def __init__(self, upstream_stream, downstream_stream):
        self.upstream_stream_ = upstream_stream
        self.downstream_stream_ = downstream_stream

    def get_downstream_stream(self):
        return self.downstream_stream_

    def get_upstream_stream(self):
        return self.upstream_stream_


class BmfNode:

    def __init__(self,
                 module_info,
                 option,
                 upstream_streams,
                 input_manager='default',
                 pre_module=None,
                 scheduler=0):
        if isinstance(module_info, dict):
            self.module_info_ = module_info
        else:
            self.module_info_ = {"name": module_info}
        self.option_ = option
        self.scheduler_ = scheduler

        # the input stream of current node and corresponding
        # output stream of upstream node point to same stream instance
        self.input_streams_ = {}
        if upstream_streams is not None:
            self.graph_ = self.init_input_streams(upstream_streams)
        else:
            from .bmf_graph import BmfGraph
            self.graph_ = BmfGraph(option)
            print('DEBUG GRAPH created new graph for empty inputstream node')

        # input manager
        self.input_manager_ = input_manager

        # pre_allocated module if exists
        self.pre_module = pre_module

        self.user_callbacks = {}

        # generate node_id and add node to graph
        assert (self.graph_ is not None), "graph is none when create node"
        self.id_ = self.graph_.generate_node_id()
        self.graph_.add_node(self)

        # output stream is empty now, created by calling stream() or []
        # TODO: do we need keep all streams or only need keep all edges?
        self.output_streams_ = {}
        self.output_stream_idx = 0
        self.output_stream_idx_mutex_ = threading.Lock()

        # downstream edges
        # while the output stream is actually connected a node, it will
        # create an edge and add to outgoing_edges_
        self.outgoing_edges_ = []

    def init_input_stream_and_edge(self, upstream_stream, notify):
        graph = None

        if upstream_stream is not None:
            # create input stream
            input_stream = BmfStream(upstream_stream.get_name(),
                                     self,
                                     notify,
                                     stream_alias=upstream_stream.get_alias())
            self.input_streams_[notify] = input_stream

            # get graph
            graph = upstream_stream.get_graph()

            # create edge
            edge = BmfEdge(upstream_stream, input_stream)

            # add edge to upstream node
            if upstream_stream.get_node() is not None:
                upstream_stream.get_node().add_outgoing_edge(edge)

        return graph

    def init_input_streams(self, upstream_streams):
        graph = None

        from .bmf_graph import BmfGraph

        if upstream_streams is None:
            return

        elif isinstance(upstream_streams, BmfGraph):
            # for source node, there is no input streams
            # use graph to initialize node
            return upstream_streams

        elif isinstance(upstream_streams, BmfStream):
            # if there is only one upstream stream, notify is 0
            graph = self.init_input_stream_and_edge(upstream_streams, 0)

        elif isinstance(upstream_streams, (list, tuple)):
            for index, upstream_stream in enumerate(upstream_streams):
                # for list input, index is notify
                graph = self.init_input_stream_and_edge(upstream_stream, index)

        elif isinstance(upstream_streams, dict):
            for (notify, upstream_stream) in upstream_streams.items():
                graph = self.init_input_stream_and_edge(
                    upstream_stream, notify)

        return graph

    def generate_stream_name(self):
        # stream name format: $(node_type)_$(node_id)_$(stream_index)
        self.output_stream_idx_mutex_.acquire()
        stream_name = self.module_info_["name"] + '_' + str(
            self.id_) + '_' + str(self.output_stream_idx)
        self.output_stream_idx += 1
        self.output_stream_idx_mutex_.release()
        return stream_name

    def stream(self, notify=None, stream_alias=None):
        if notify is None:
            notify = 0

        if notify not in self.output_streams_.keys():
            stream_name = self.generate_stream_name()

            # create output stream
            s = BmfStream(stream_name, self, notify, stream_alias=stream_alias)

            self.output_streams_[notify] = s

        return self.output_streams_[notify]

    def __getitem__(self, item):
        return self.stream(notify=item)

    def add_outgoing_edge(self, edge):
        if edge is not None:
            self.outgoing_edges_.append(edge)

    def get_outgoing_edges(self):
        return self.outgoing_edges_

    def get_input_streams(self):
        return self.input_streams_

    def get_output_streams(self):
        return self.output_streams_

    def get_module_info(self):
        return self.module_info_

    def get_id(self):
        return self.id_

    def get_scheduler(self):
        return self.scheduler_

    def set_scheduler(self, schediler):
        self.scheduler_ = schediler

    def get_option(self):
        return self.option_

    def get_pre_module(self):
        return self.pre_module

    def get_graph(self):
        return self.graph_

    def get_input_manager(self):
        return self.input_manager_

    def set_input_manager(self, input_manager):
        self.input_manager_ = input_manager

    def run(self):
        self.graph_.run()

    def add_user_callback(self, key, cb):
        from bmf.lib._bmf import engine
        callback = engine.Callback(cb)
        self.user_callbacks[key] = (callback.uid(), callback)

    def get_user_callback(self):
        return self.user_callbacks

    def start(self):
        print('no output stream')

    def create_sync_module(self):
        from bmf.lib._bmf import engine
        node_option = json.dumps(self.option_)

        # convert node option for filter
        if self.module_info_["name"] == "c_ffmpeg_filter":
            node_config = self.get_graph().generate_node_config(self)
            node_option = engine.convert_filter_para(node_config.dump())

        # create module
        mod = engine.Module(self.module_info_["name"], node_option,
                            self.module_info_["type"],
                            self.module_info_["path"],
                            self.module_info_["entry"])

        # input stream list
        input_stream_id = []
        if self.module_info_[
                "name"] == "c_ffmpeg_encoder" and 1 in self.get_input_streams(
                ).keys():
            input_stream_id.append(0)
            input_stream_id.append(1)
        else:
            for id in self.get_input_streams().keys():
                input_stream_id.append(id)

        # output stream list
        output_stream_id = []
        if self.module_info_["name"] == "c_ffmpeg_decoder":
            for key in self.get_output_streams().keys():
                if key == "video":
                    output_stream_id.append(0)
                elif key == "audio":
                    output_stream_id.append(1)
        else:
            for id in self.get_output_streams().keys():
                output_stream_id.append(id)

        # create sync module
        sync_module = SyncModule(mod, input_stream_id, output_stream_id)
        return sync_module
