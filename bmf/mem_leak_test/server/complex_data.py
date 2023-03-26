import numpy as np
import time
from bmf import Module, Log, LogLevel, InputType, ProcessResult, Packet, Timestamp, scale_av_pts, av_time_base, \
    BmfCallBackType, VideoFrame, AudioFrame
import base64
import sys

if sys.version_info.major == 2:
    from Queue import Queue
else:
    from queue import Queue


class complex_data(Module):
    def process(self, task):
        for (input_id, input_queue) in task.get_inputs().items():
            while not input_queue.empty():
                pkt = input_queue.get()

                if pkt.get_timestamp() == Timestamp.EOF:
                    for key in task.get_outputs():
                        task.get_outputs()[key].put(Packet.generate_eof_packet())
                    task.set_timestamp(Timestamp.DONE)
                    return

                if pkt.get_data() is not None:
                    in_data = pkt.get_data()
                    video_data = in_data["video_data"]
                    pkt_video_data = Packet()
                    pkt_video_data.set_timestamp(pkt.get_timestamp())
                    pkt_video_data.set_data(video_data)
                    task.get_outputs()[0].put(pkt_video_data)

                    if len(task.get_outputs()) > 1:
                        extra_data = in_data["extra_data"]
                        pkt_extra_data = Packet()
                        pkt_extra_data.set_timestamp(pkt.get_timestamp())
                        pkt_extra_data.set_data(extra_data)
                        task.get_outputs()[1].put(pkt_extra_data)

        return ProcessResult.OK


if __name__ == '__main__':
    pass
