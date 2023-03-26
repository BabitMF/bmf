import pygame

from bmf import Module, Log, LogLevel, InputType, ProcessResult, Packet, Timestamp, scale_av_pts, av_time_base, \
    BmfCallBackType, VideoFrame, AudioFrame

'''
Option example:
    option = {
        'Type': 'text',
        'Text': 'Byte Dance',
        'StartTime': 0,
        'Duration': 5,
        'Position': {
            'PosX': '50%',
            'PosY': '50%',
            'Width': '20%',
            'Height': '20%'
        },
        'FontType': 'SimHei',
        'FontSize': 23,
        'FontColor': '0xFFFFFF00',
        'BackgroundColor': '0xFFFFFF00',
        'ShadowColor': '0xFFFFFF00',
        'HorizontalAlign': 1,
        'VerticalAlign': 0,
        'MultiLine': 0,
        'LineSpace': 1.5,
        'ReplaceSuffix': 0,
        'Animation': {
            'Type': 'X',
            'Speed': 'default',
            'Duration': 3
        },
        'Italic': 1,
        'FontWeight': 'bold',
        'Underline': 1
    }
'''


class text_to_image(Module):
    def __init__(self, node, option=None):
        self.node_ = node

        if 'Text' not in option:
            raise Exception('no text exists')
        self.text = option['Text']

        self.local_path_ = option['local_path']

        self.option = option

        # init pygame
        pygame.init()

    def process(self, task):
        # turn text to image
        text = self.option['Text']
        font = pygame.font.SysFont(self.option['FontType'], self.option['FontSize'])
        ftext = font.render(text, True, (0, 0, 0), (255, 255, 255))

        # save image
        pygame.image.save(ftext, self.local_path_)

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
