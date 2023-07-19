import sys
import threading
from bmf import Packet
from ..builder import create_module, GraphMode
from ..python_sdk import Timestamp


class ServerGateway:

    def __init__(self, server_graph):
        self.graph = server_graph
        self.ready_close = False

        # id of all coming works
        self.count_id = 0
        # id of done works
        self.result_id = 0

        self.status_mutex_ = threading.RLock()

        # relationship of work and result
        self.result_dict = {}
        # relationship of work and event
        self.event_dict = {}

        # graph_finish event
        self.finish_event = threading.Event()

    def init(self):
        # start a monitor thread, monitor the result packets in graph output_stream
        monitor_thread_ = threading.Thread(target=self.monitor_thread)
        monitor_thread_.setDaemon(True)
        monitor_thread_.start()

        # run as server_mode
        self.graph.run(streams=self.graph.node_streams_, mode=GraphMode.SERVER)

    def monitor_thread(self):
        # name of graph output_stream
        # TODO: currently, we think graph has only one output_stream, which is able to expand to more
        graph_output_stream = self.graph.node_streams_[0].get_name()

        while True:
            # only when no more works will come and all the results have been achieved, we can quit this thread
            # and we allow graph to close right now
            if self.ready_close and self.result_id == self.count_id:
                self.finish_event.set()
                break

            # try to get pkt from graph_output_stream
            pkt = self.graph.poll_packet(graph_output_stream, block=True)

            # currently, we think each work will produce only one result packet, which contains return value in its data
            if pkt is not None and pkt.defined():
                if pkt.class_name == "std::string":
                    # one more access result
                    self.result_id += 1
                    # save this result
                    self.result_dict[self.result_id] = pkt.get(str)
                    # allow process_work thread to return the result
                    self.event_dict[self.result_id].set()

    def process_work(self, pkt):
        # use Lock to avoid concurrency process
        self.status_mutex_.acquire()

        # one more coming work
        self.count_id += 1

        # if monitor_thread already quit, close this thread at once
        if self.finish_event.is_set():
            self.status_mutex_.release()
            return None

        # work_id represents each work
        work_id = self.count_id

        # push pkt into graph_input
        graph_input_name = self.graph.input_streams_[0].get_name()
        self.graph.fill_packet(graph_input_name, pkt)

        # generate an EOF pkt and push into graph_input
        eof_packet = Packet.generate_eof_packet()
        self.graph.fill_packet(graph_input_name, eof_packet)

        # result that will be returned
        process_result = {}
        # correspond work_id with its result, record them in result_dict
        self.result_dict[work_id] = process_result

        # event that waiting for the result of work
        return_event = threading.Event()
        # correspond work_id with its event, record them in event_dict
        self.event_dict[work_id] = return_event

        self.status_mutex_.release()

        # wait until result of this work is achieved
        return_event.wait()

        # return result
        process_result = self.result_dict[work_id]
        return process_result

    def close(self):
        # indicate that client is willing to close graph, no more work will come
        self.ready_close = True

        # wait util all work done and all result got
        if self.result_id != self.count_id:
            self.finish_event.wait()

        # generate an eos pkt and push into graph_input
        eos_pkt = Packet.generate_eos_packet()
        graph_input_name = self.graph.input_streams_[0].get_name()
        self.graph.fill_packet(graph_input_name, eos_pkt)

        # close graph
        self.graph.close()
