import abc
from .module import Module, ProcessResult
from .utils import Log, LogLevel
from .timestamp import Timestamp


class SubGraph(Module):

    @abc.abstractmethod
    def create_graph(self, option=None):
        # use self.graph to build a processing graph
        # put name of input streams into self.inputs
        pass

    def get_graph_config(self):
        return self.graph.get_graph_config()

    def is_subgraph(self):
        return True

    def finish_create_graph(self, output_streams):
        from bmf import GraphMode
        self.graph.mode = GraphMode.SUBGRAPH
        self.graph.parse_output_streams(output_streams)
        self.graph.graph_config_, _ = self.graph.generate_graph_config()

    def __init__(self, node_id, option=None):
        if option is None:
            option = {}
        self.dump_graph_ = 0
        if 'dump_graph' in option.keys():
            self.dump_graph_ = option['dump_graph']

        # construct graph in init function
        # create a bmf graph
        from ..builder import bmf
        self.graph = bmf.graph({
            'dump_graph': self.dump_graph_,
            'graph_name': 'subgraph_node_%d' % (node_id)
        })

        self.inputs = []

        # record if output stream is done
        self.stream_done = {}
        self.output_streams = None

        self.node_id_ = node_id
        self.option_ = option

        self.create_graph(option)

    def process(self, task):
        if self.graph is None:
            return

        # process input
        for (index, input_queue) in task.get_inputs().items():
            while self.graph is not None and not input_queue.empty():
                pkt = input_queue.get()
                # don't need to process empty packet
                if pkt.get_timestamp() != Timestamp.UNSET:
                    # receive eof, fill an eof packet to sub graph
                    if pkt.get_timestamp() == Timestamp.EOF:
                        self.graph.fill_eof(self.inputs[index])
                        Log.log_node(LogLevel.DEBUG, self.node_id_, 'fill eof',
                                     'on input', index)
                        break

                    # fill normal packet to sub graph
                    self.graph.fill_packet(self.inputs[index], pkt)
                    Log.log_node(LogLevel.DEBUG, self.node_id_, 'fill packet',
                                 pkt.get_data(), 'time', pkt.get_timestamp(),
                                 'on input', index)

        # process output
        for (i, stream) in enumerate(self.output_streams):
            output_queue = task.get_outputs()[i]
            while self.graph is not None:
                # poll a packet from sub graph
                output_pkt = self.graph.poll_packet(stream)
                if output_pkt is not None and output_pkt.defined():
                    # add the packet to output queue to let outside graph process
                    output_queue.put(output_pkt)
                    Log.log_node(LogLevel.DEBUG, self.node_id_, 'output', i,
                                 'send packet', output_pkt.get_data(), 'time',
                                 output_pkt.get_timestamp())
                    if output_pkt.get_timestamp() == Timestamp.EOF:
                        self.stream_done[i] = 1
                else:
                    break

        # all output streams done, close sub graph
        if len(self.stream_done) == len(self.output_streams):
            # consider that sub-graph could be made of some infinity nodes
            # force to close sub graph
            Log.log_node(LogLevel.DEBUG, self.node_id_,
                         'start close sub-graph')
            self.graph.force_close()
            self.graph = None

            # notify downstream node closed
            task.set_timestamp(Timestamp.DONE)
            for (i, _) in enumerate(self.output_streams):
                output_queue = task.get_outputs()[i]
                output_queue.put(Packet.generate_eof_packet())
            Log.log_node(LogLevel.DEBUG, self.node_id_, 'sub-graph closed')

        return ProcessResult.OK

    def close(self):
        if self.graph is not None:
            Log.log_node(LogLevel.DEBUG, self.node_id_,
                         'sub-graph force closed')
            self.graph.force_close()
            self.graph = None
