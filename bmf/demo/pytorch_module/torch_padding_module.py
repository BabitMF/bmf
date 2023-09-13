from bmf import *
import bmf.hml.hmp as mp

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T

class torch_padding_module(Module):
    def __init__(self, node, option=None):
        self.node_ = node
        self.option_ = option
        self.eof_received_ = False

        self.padding = [0]
        self.fill = 0
        self.mode = 'constant'
        valid_modes = ['constant', 'edge', 'reflect', 'symmetric']
        if 'padding' in option.keys():
            self.padding = list(map(int, option['padding'].split(',')))
        if 'fill' in option.keys():
            self.fill = option['fill']
        if 'mode' in option.keys():
            self.mode = option['mode']
            if self.mode not in valid_modes:
                Log.log(LogLevel.ERROR, "Invalid fill modes, choose from 'constant', 'edge', 'reflect', 'symmetric'")
                return
        
    def process(self, task):
        # get input and output packet queue
        input_queue = task.get_inputs()[0]
        output_queue = task.get_outputs()[0]

        # add all input frames into frame cache
        while not input_queue.empty():
            in_pkt = input_queue.get()

            if in_pkt.timestamp == Timestamp.EOF:
                # we should done all frames processing in following loop
                self.eof_received_ = True
                continue

            in_frame = in_pkt.get(VideoFrame)

            if (in_frame.frame().device() == mp.Device('cpu')):
                in_frame = in_frame.cuda()

            padded = []
            for t in in_frame.frame().data():
                padded.append(T.functional.pad(torch.from_dlpack(t).permute(2,0,1), self.padding, self.fill, self.mode).permute(1,2,0).contiguous())

            # import pdb
            # pdb.set_trace()
            videoframe_out = VideoFrame(mp.Frame(mp.from_dlpack(padded), in_frame.frame().pix_info()))
            md = sdk.MediaDesc()
            md.pixel_format(mp.kPF_NV12).device(mp.Device("cuda:0"))
            videoframe_out = sdk.bmf_convert(videoframe_out, sdk.MediaDesc(), md)
            videoframe_out.pts = in_frame.pts
            videoframe_out.time_base = in_frame.time_base
            out_pkt = Packet(videoframe_out)
            out_pkt.timestamp = videoframe_out.pts
            output_queue.put(out_pkt)

        if self.eof_received_:
            output_queue.put(Packet.generate_eof_packet())
            Log.log_node(LogLevel.DEBUG, self.node_, 'output stream', 'done')
            task.set_timestamp(Timestamp.DONE)

        return ProcessResult.OK
