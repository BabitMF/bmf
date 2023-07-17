import os
import threading
from bmf import Log, LogLevel
from bmf import Packet
from ..builder import GraphMode
from ..python_sdk import Timestamp


class ServerGatewayNew:

    def __init__(self, server_graph):
        self.graph = server_graph

        # id of all coming jobs
        self.count_id = 0
        # id of done jobs
        self.result_id = 0
        # id of front job
        self.front_id = 0

        # dict of job result
        self.result_dict = {}
        # dict of job name
        self.alias_dict = {}

        # closing flag
        self.closed = False

        # block event
        self.block_event = threading.Event()

    def init(self):
        # start a polling thread, consistently trying to fetch job results from graph output
        monitor_thread_ = threading.Thread(target=self.polling_job_result)
        monitor_thread_.setDaemon(True)
        monitor_thread_.start()

        # run as server mode
        self.graph.run(streams=self.graph.node_streams_, mode=GraphMode.SERVER)

    def process_work(self, pkt, name=None):
        # one more coming job
        self.count_id += 1

        # record job name
        if name is not None:
            self.alias_dict[self.count_id] = name

        # push pkt into graph_input
        graph_input_name = self.graph.input_streams_[0].get_name()
        self.graph.fill_packet(graph_input_name, pkt)

        # generate an EOF pkt and push into graph_input
        eof_packet = Packet.generate_eof_packet()
        self.graph.fill_packet(graph_input_name, eof_packet)

    def polling_job_result(self):
        graph_output_stream = self.graph.node_streams_[0].get_name()

        while True:
            if self.result_id == self.count_id and self.closed:
                break

            # try to get pkt from graph_output_stream
            pkt = self.graph.poll_packet(graph_output_stream, block=True)

            # currently, we think each job will produce only one result packet, which contains return value in its data
            if pkt is not None and pkt.defined():
                # one more access result
                if pkt.class_name == "std::string":
                    self.result_id += 1
                    # get result
                    if self.result_id in self.alias_dict:
                        res_name = self.alias_dict[self.result_id]
                    else:
                        res_name = "res_" + str(self.result_id)
                    # save result
                    self.result_dict[res_name] = pkt.get(str)
                    if not self.block_event.is_set():
                        self.block_event.set()
                        self.block_event.clear()

    def close(self):
        # generate an eos pkt and push into graph_input
        eos_pkt = Packet.generate_eos_packet()
        graph_input_name = self.graph.input_streams_[0].get_name()
        self.graph.fill_packet(graph_input_name, eos_pkt)

        # set close flag
        self.closed = True

        # close bmf graph
        self.graph.close()

    # waiting until all jobs done and return all results at one time
    def request_for_res(self):
        # request for all results
        while len(self.result_dict) < self.count_id:
            self.block_event.wait()
        # return results
        return self.result_dict

    # return result of specified job according to job name
    # supports two conditions:
    # if block is False, immediately return even if result is empty
    # if block is True, wait until the specified job done and then return its result, regardless of other jobs
    def get_by_job_name(self, name, block=False):
        if name not in self.alias_dict.values():
            Log.log(LogLevel.ERROR, "incorrect job name")
            os._exit(1)
        if name not in self.result_dict.keys():
            if not block:
                return None
            else:
                while (name not in self.result_dict.keys()):
                    self.block_event.wait()
                return {name: self.result_dict[name]}
        else:
            return {name: self.result_dict[name]}

    # judge if result is empty
    def empty_result(self):
        if self.front_id >= self.count_id:
            return True
        return False

    # return the top result of output queue
    def get_front_result(self):
        self.front_id += 1

        if self.front_id > self.count_id:
            Log.log(LogLevel.ERROR, "result queue is empty")
            os._exit(1)

        if self.front_id in self.alias_dict:
            res_name = self.alias_dict[self.front_id]
        else:
            res_name = "res_" + str(self.front_id)

        while (res_name not in self.result_dict.keys()):
            self.block_event.wait()

        res_value = self.result_dict[res_name]
        del self.result_dict[res_name]
        return {res_name: res_value}
