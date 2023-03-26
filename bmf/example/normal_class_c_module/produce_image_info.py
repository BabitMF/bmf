import numpy as np
import time
from bmf import Module, Log, LogLevel, InputType, ProcessResult, Packet, Timestamp, scale_av_pts, av_time_base, \
    BmfCallBackType, VideoFrame, AudioFrame
import sys

if sys.version_info.major == 2:
    from Queue import Queue
else:
    from queue import Queue


class Rect(object):
    def __init__(self, x=0, y=0, width=0, height=0):
        self.x = x
        self.y = y
        self.width = width
        self.height = height


class ImageInfo(object):
    def __init__(self, image_info="hello world", width=200, height=100, rect=Rect()):
        self.image_info = image_info
        self.width = width
        self.height = height
        self.rect = rect


class produce_image_info(Module):
    def __init__(self, node=None, option=None):
        self.node_ = node
        self.option_ = option
        Log.log_node(LogLevel.DEBUG, self.node_, "init produce_image_info")
        if option is None:
            Log.log(LogLevel.ERROR, "Option is none")
            return
        self.num_ = 100
        if 'num' in option.keys():
            self.num_ = option['num']
        self.timestamp_ = 0

    def reset(self):
        # clear status
        self.eof_received_ = False

    def process(self, task):
        # get input and output packet queue
        output_queue = task.get_outputs()[0]
        out_pkt = Packet()
        out_pkt.set_data(ImageInfo())
        out_pkt.set_timestamp(self.timestamp_)
        self.timestamp_ = self.timestamp_ + 1
        output_queue.put(out_pkt)
        # add all input frames into frame cache
        time.sleep(0.1)
        if self.timestamp_ == self.num_:
            output_queue.put(Packet.generate_eof_packet())
            Log.log_node(LogLevel.DEBUG, self.node_,
                         'output stream', 'done')
            task.set_timestamp(Timestamp.DONE)

        return ProcessResult.OK
