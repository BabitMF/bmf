
import os
import sys
from bmf import Module, Log, LogLevel, InputType, ProcessResult, Packet, Timestamp, VideoFrame, av_time_base

class callback_module(Module):
    def __init__(self, node=None, option=None):
        self.node_ = node
        self.pkt_num_ = 0
        self.last_input_num_ = 0
        self.last_output_num_ = 0
        self.is_eof_ = False

    def process(self, task):
        if len(task.get_inputs()) != self.last_input_num_:
            print("Node:", self.node_, "The input stream number changed from ",
                  self.last_input_num_, " to ", len(task.get_inputs()))
            self.last_input_num_ = len(task.get_inputs())
        if len(task.get_outputs()) != self.last_output_num_:
            print("Node:", self.node_, "The output stream number changed from ",
                  self.last_output_num_, " to ", len(task.get_outputs()))
            self.last_output_num_ = len(task.get_outputs())

        for (input_id, input_packets) in task.get_inputs().items():
            output_packets = None
            if len(task.get_outputs()) > 0 and input_id < len(task.get_outputs()):
                output_packets = task.get_outputs()[input_id]
            while not input_packets.empty():
                pkt = input_packets.get()
                if pkt.timestamp == Timestamp.DYN_EOS:
                    print("got DYN_EOS message")
                if pkt.timestamp == Timestamp.EOF:
                    Log.log_node(LogLevel.DEBUG, task.get_node(), "Receive EOF")
                    if output_packets is not None:
                        output_packets.put(Packet.generate_eof_packet())
                    task.timestamp = Timestamp.DONE
                elif pkt.timestamp != Timestamp.UNSET:
                    Log.log_node(LogLevel.DEBUG, task.get_node(), ' pkt processed number is: ',
                                 self.pkt_num_)
                    if output_packets is not None:
                        output_packets.put(pkt)

                    self.pkt_num_ += 1

                    # dynamical add encoder is not a good example to show add action since the init of encoder
                    # some time will not be inited correctly and will generate some encode errors, won't use
                    # encoder temporary
                    #if self.pkt_num_ == 50:
                    #    #dynmaic add notice to the graph
                    #    if self.callback_ is not None:
                    #        self.callback_(0, {'add_e': 'encoder1'})
                    
                    if self.pkt_num_ == 80:
                        #dynmaic add notice to the graph
                        if self.callback_ is not None:
                            message = "{\"add_d\": \"decoder1\"}"
                            self.callback_(0, bytes(message, "utf-8"))
                    if self.pkt_num_ == 90:
                        #dynmaic add notice to the graph
                        if self.callback_ is not None:
                            message = "{\"remove\": \"decoder1\"}"
                            self.callback_(0, bytes(message, "utf-8"))

        return ProcessResult.OK

if __name__ == '__main__':
    module = callback_module()
