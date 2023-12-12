import sys
import random
from typing import List, Optional
import pdb

from bmf import *
import bmf.hml.hmp as mp

class text_module(Module):
    def __init__(self, node, option=None):
        self.node_ = node
        self.eof_received_ = False
        self.prompt_path = './prompt.txt'
        if 'path' in option.keys():
            self.prompt_path = option['path']

    def process(self, task):
        # pdb.set_trace()
        input_packets = task.get_inputs()[0]
        output_queue = task.get_outputs()[0]

        while not input_packets.empty():
            pkt = input_packets.get()
            if pkt.timestamp == Timestamp.EOF:
                output_queue.put(Packet.generate_eof_packet())
                Log.log_node(LogLevel.DEBUG, self.node_, 'output text stream', 'done')
                task.set_timestamp(Timestamp.DONE)
                return ProcessResult.OK

            # if self.eof_received_ == True:
            #     output_queue.put(Packet.generate_eof_packet())
            #     Log.log_node(LogLevel.DEBUG, self.node_, 'output text stream', 'done')
            #     task.set_timestamp(Timestamp.DONE)
            #     return ProcessResult.OK

            prompt_dict = dict()
            with open(self.prompt_path) as f:
                for line in f:
                    pk, pt = line.partition(":")[::2]
                    prompt_dict[pk] = pt

            out_pkt = Packet(prompt_dict)
            out_pkt.timestamp = 0
            output_queue.put(out_pkt)
            # self.eof_received_ = True

        return ProcessResult.OK

def register_inpaint_module_info(info):
    info.module_description = "Text file IO module"