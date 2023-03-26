import bmf
from bmf import Module, Log, LogLevel, InputType, ProcessResult, Packet, Timestamp, scale_av_pts, av_time_base, \
    BmfCallBackType, VideoFrame, AudioFrame
import os

'''
Option example:
    option = {
        'segment_duration': 5,
        'local_path': '../files/xxx.png'
    }
'''


class background_audio(Module):
    def __init__(self, node, option=None):
        self.node_ = node

        if 'segment_duration' not in option:
            raise Exception('no segment duration')
        self.duration = option['segment_duration']

        if 'local_path' not in option:
            raise Exception('no local path')
        self.local_path_ = option['local_path']

    def process(self, task):
        # create local silent audio
        os.system('ffmpeg -f lavfi -t ' + str(self.duration) + ' -i anullsrc ' + str(self.local_path_) + ' -y')

        # fill input info
        input_info = {
            "input_path": self.local_path_
        }

        # prepare output data
        input_info_list = []
        input_info_list.append(input_info)
        data = {'input_path': input_info_list}

        # prepare output packet
        pkt = Packet()
        pkt.set_timestamp(1)
        pkt.set_data(data)

        # add to output queue, also add eof
        output_queue = task.get_outputs()[0]
        output_queue.put(pkt)
        output_queue.put(Packet.generate_eof_packet())

        # module process done
        task.set_timestamp(Timestamp.DONE)

        return ProcessResult.OK
