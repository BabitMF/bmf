import time

import sys

sys.path.append("../../..")
from bmf import Module, Log, LogLevel, InputType, ProcessResult, Packet, Timestamp, scale_av_pts, av_time_base, \
    BmfCallBackType
import sys

if sys.version_info.major == 2:
    from Queue import Queue
else:
    from queue import Queue

'''
perform as a analysis module for self_test
'''


class analysis(Module):
    def __init__(self, node, option=None):
        self.node_ = node

        # need to reset when using the module
        self.task_count = 0

        if option is None:
            Log.log_node(LogLevel.ERROR, self.node_, "Option is none")
            return

        for i in range(3):
            time.sleep(1)
            Log.log(LogLevel.ERROR, "Waiting 3 seconds to init the module")

    # reset the module
    def reset(self):
        self.task_count += 1
        Log.log(LogLevel.ERROR, "task_count is : " + str(self.task_count))

    def process(self, task=None):

        for (input_id, input_packets) in task.get_inputs().items():
            # output queue
            output_packets = task.get_outputs()[input_id]
            while not input_packets.empty():
                pkt = input_packets.get()
                if pkt.timestamp == Timestamp.EOF:
                    Log.log_node(LogLevel.DEBUG, task.get_node(), "Receive EOF")
                    output_packets.put(Packet.generate_eof_packet())
                    task.timestamp = Timestamp.DONE
                if pkt.timestamp != Timestamp.UNSET and pkt.defined():
                    output_packets.put(pkt)
                    Log.log_node(LogLevel.DEBUG, task.get_node(),
                                 "process input", input_id, 'packet',
                                 output_packets.front().timestamp)
        return ProcessResult.OK
