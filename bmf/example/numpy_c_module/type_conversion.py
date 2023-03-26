import numpy as np
import time
from bmf import Module, Log, LogLevel, InputType, ProcessResult, Packet, Timestamp, scale_av_pts, av_time_base, \
    BmfCallBackType, VideoFrame, AudioFrame
import sys

if sys.version_info.major == 2:
    from Queue import Queue
else:
    from queue import Queue
from fractions import Fraction


class type_conversion(Module):
    def __init__(self, node=None, option=None):
        self.node_ = node
        self.option_ = option
        if option is None:
            Log.log(LogLevel.ERROR, "Option is none")
            return

        # to_numpy
        if 'to_numpy' in option.keys():
            self.trans_to_numpy_ = option['to_numpy']

        start_time = time.time()
        self.eof_received_ = False
        self.pts_ = 0

    def reset(self):
        # clear status
        self.eof_received_ = False

    def process(self, task):
        # get input and output packet queue
        input_queue = task.get_inputs()[0]
        output_queue = task.get_outputs()[0]

        # add all input frames into frame cache
        while not input_queue.empty():
            in_pkt = input_queue.get()

            if in_pkt.get_timestamp() == Timestamp.EOF:
                # we should done all frames processing in following loop
                self.eof_received_ = True
                continue

            out_pkt = Packet()
            if (self.trans_to_numpy_ == 1):
                out_pkt.set_data(in_pkt.get_data().to_ndarray())
                # print(in_pkt.get_data().pts)
                # print(in_pkt.get_data().time_base)
            else:
                # print(in_pkt.get_data())
                frame = VideoFrame.from_ndarray(in_pkt.get_data(), "yuv420p")
                frame.pts = self.pts_
                self.pts_ = self.pts_ + 1001
                frame.time_base = Fraction(1, 30000)
                out_pkt.set_data(frame)

            out_pkt.set_timestamp(in_pkt.get_timestamp())
            output_queue.put(out_pkt)

        # add eof packet to output
        if self.eof_received_:
            output_queue.put(Packet.generate_eof_packet())
            Log.log_node(LogLevel.DEBUG, self.node_,
                         'output stream', 'done')
            task.set_timestamp(Timestamp.DONE)

        return ProcessResult.OK
