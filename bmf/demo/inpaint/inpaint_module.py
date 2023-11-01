import sys
import random
from typing import List, Optional
import pdb

from bmf import *
import bmf.hml.hmp as mp

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T

sys.path.append('./MAT')
from MAT import dnnlib
from MAT import legacy
from MAT.datasets.mask_generator_512 import RandomMask
from MAT.networks.mat import Generator

def named_params_and_buffers(module):
    assert isinstance(module, torch.nn.Module)
    return list(module.named_parameters()) + list(module.named_buffers())

def copy_params_and_buffers(src_module, dst_module, require_all=False):
    assert isinstance(src_module, torch.nn.Module)
    assert isinstance(dst_module, torch.nn.Module)
    src_tensors = {name: tensor for name, tensor in named_params_and_buffers(src_module)}
    for name, tensor in named_params_and_buffers(dst_module):
        assert (name in src_tensors) or (not require_all)
        if name in src_tensors:
            tensor.copy_(src_tensors[name].detach()).requires_grad_(tensor.requires_grad)

class inpaint_module(Module):
    def __init__(self, node, option=None):
        # generate random seed
        seed = 240  # pick up a random number
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        
        self.node_ = node
        self.eof_received_ = False
        network_pkl = './MAT/CelebA-HQ_512.pkl'
        self.trunc = 1.0
        self.noise_mode = 'const'
        if 'network' in option.keys():
            network_pkl = option['network']
        if 'trunc' in option.keys():
            self.trunc = option['trunc']
        if 'noise-mode' in option.keys():
            self.noise_mode = option['noise-mode']
        device = torch.device('cuda')

        with dnnlib.util.open_url(network_pkl) as f:
            G_saved = legacy.load_network_pkl(f)['G_ema'].to(device).eval().requires_grad_(False)
        net_res = 512
        self.G = Generator(z_dim=512, c_dim=0, w_dim=512, img_resolution=net_res, img_channels=3).to(device).eval().requires_grad_(False)
        copy_params_and_buffers(G_saved, self.G, require_all=True)
        self.labels = torch.zeros([1, self.G.c_dim], device=device)


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

            infer_t = (torch.from_dlpack(in_frame.frame().data()[0]).float().permute(2, 0, 1) / 127.5 - 1).unsqueeze(0)
            z = torch.from_numpy(np.random.randn(1, self.G.z_dim)).to('cuda')
            for k in range(1, 72):
                rot_img = T.functional.rotate(infer_t, 5 * k)
                rot_mask = (rot_img.sum(1, keepdim=True) != 0).type(torch.float)
                output = self.G(rot_img, rot_mask, z, self.labels, truncation_psi=self.trunc, noise_mode=self.noise_mode)
                output = (output.permute(0, 2, 3, 1) * 127.5 + 127.5).round().clamp(0, 255).to(torch.uint8)

                videoframe_out = VideoFrame(mp.Frame(mp.from_dlpack(output.squeeze(0).contiguous()), in_frame.frame().pix_info()))
                md = sdk.MediaDesc()
                md.pixel_format(mp.kPF_NV12).device(mp.Device("cuda:0"))
                videoframe_out = sdk.bmf_convert(videoframe_out, sdk.MediaDesc(), md)
                videoframe_out.pts = in_frame.pts + k - 1
                videoframe_out.time_base = in_frame.time_base
                out_pkt = Packet(videoframe_out)
                out_pkt.timestamp = videoframe_out.pts
                output_queue.put(out_pkt)

        if self.eof_received_:
            output_queue.put(Packet.generate_eof_packet())
            Log.log_node(LogLevel.DEBUG, self.node_, 'output stream', 'done')
            task.set_timestamp(Timestamp.DONE)

        return ProcessResult.OK

def register_inpaint_module_info(info):
    info.module_description = "Image inpaint module using MAT model"
    info.module_tag = ModuleTag.TAG_DEVICE_HWACCEL
