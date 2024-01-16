import sys
import random
from typing import List, Optional
import numpy as np
import pdb

from bmf import *
import bmf.hml.hmp as mp
sys.path.append('./controlnet')
from canny2image_TRT import hackathon

class controlnet_module(Module):
    def __init__(self, node, option=None):
        self.node_ = node
        self.eof_received_ = False
        self.hk = hackathon()
        self.hk.initialize()
        self.prompt_path = './prompt.txt'
        self.eof_received_ = [False, False]
        self.prompt_ = None
        self.frame_list_ = []
        if 'path' in option.keys():
            self.prompt_path = option['path']

    def process(self, task):
        img_queue = task.get_inputs()[0]
        pmt_queue = task.get_inputs()[1]
        output_queue = task.get_outputs()[0]

        while not pmt_queue.empty():
            pmt_pkt = pmt_queue.get()

            if pmt_pkt.timestamp == Timestamp.EOF:
                self.eof_received_[0] = True
            else:
                pmt = pmt_pkt.get(dict)
                self.prompt_ = pmt

        while not img_queue.empty():
            in_pkt = img_queue.get()

            if in_pkt.timestamp == Timestamp.EOF:
                self.eof_received_[1] = True
            else:
                self.frame_list_.append(in_pkt.get(VideoFrame))

        while self.prompt_ and len(self.frame_list_) > 0:
            in_frame = self.frame_list_[0]
            del self.frame_list_[0]
            
            gen_img = self.hk.process(in_frame.cpu().frame().data()[0].numpy(),
                pmt['prompt'], pmt['a_prompt'], pmt['n_prompt'],
                1,
                256,
                20,
                False,
                1,
                9,
                2946901,
                0.0,
                100,
                200)

            rgbinfo = mp.PixelInfo(mp.PixelFormat.kPF_RGB24,
                                    in_frame.frame().pix_info().space,
                                    in_frame.frame().pix_info().range)
            out_f = mp.Frame(mp.from_numpy(gen_img[0]), rgbinfo)
            out_vf = VideoFrame(out_f)
            out_vf.pts = in_frame.pts
            out_vf.time_base = in_frame.time_base
            out_pkt = Packet(out_vf)
            out_pkt.timestamp = out_vf.pts
            output_queue.put(out_pkt)

        if self.eof_received_[0] and self.eof_received_[1] and len(self.frame_list_) == 0:
            output_queue.put(Packet.generate_eof_packet())
            Log.log_node(LogLevel.DEBUG, self.node_, 'output text stream', 'done')
            task.set_timestamp(Timestamp.DONE)
            return ProcessResult.OK

        return ProcessResult.OK

def register_inpaint_module_info(info):
    info.module_description = "ControlNet inference module"
