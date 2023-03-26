from PIL import Image

from bmf import Module, Log, LogLevel, InputType, ProcessResult, Packet, Timestamp, scale_av_pts, av_time_base, \
    BmfCallBackType, VideoFrame, AudioFrame

'''
Option example:
    option = {
        'width': 1280,
        'height': 720,
        'background_color': '0xFFFF00FF',
        'background_image_file': '../files/xxx.png'
    }
'''


class background_image(Module):
    def __init__(self, node, option=None):
        self.node_ = node

        if 'background_color' not in option:
            raise Exception('no background color')
        self.color_tuple = self.get_color_tuple(option['background_color'])

        if 'width' not in option or 'height' not in option:
            raise Exception('no output size')
        self.size_tuple = (option['width'], option['height'])

        if 'local_path' not in option:
            raise Exception('no local path')
        self.local_path_ = option['local_path']

    def get_color_tuple(self, color_str):
        # transparency
        alpha = int(color_str[-8:-6], 16)

        # color params
        blue = int(color_str[-6:-4], 16)
        green = int(color_str[-4:-2], 16)
        red = int(color_str[-2:], 16)

        # return color-tuple in correct sequence
        return red, green, blue

    def process(self, task):
        # generate background image
        img = Image.new('RGB', self.size_tuple, self.color_tuple)

        # save background image
        img.save(self.local_path_)

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
