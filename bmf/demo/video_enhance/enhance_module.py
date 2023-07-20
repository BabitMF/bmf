#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This code is based on the implementation from https://github.com/xinntao/Real-ESRGAN

from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.download_util import load_file_from_url

from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact

from bmf import Module, Log, Timestamp, ProcessResult, LogLevel, Packet, VideoFrame
from bmf.lib._bmf.sdk import ffmpeg

from bmf.hml import hmp as mp
import numpy as np

import os


def load_model():
    model = SRVGGNetCompact(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_conv=16,
        upscale=4,
        act_type="prelu",
    )
    netscale = 4
    file_url = [
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth"
    ]
    return model, netscale, file_url


def prepare_model(model_name, file_url):
    model_path = os.path.join("weights", model_name + ".pth")
    if not os.path.isfile(model_path):
        ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
        for url in file_url:
            # model_path will be updated
            model_path = load_file_from_url(
                url=url,
                model_dir=os.path.join(ROOT_DIR, "weights"),
                progress=True,
                file_name=None,
            )
    return model_path


class EnhanceModule(Module):

    def __init__(self, node=None, option=None):
        self._node = node
        if not option:
            Log.log_node(LogLevel.ERROR, self._node, "no option")
            return

        tile = option.get("tile", 0)
        tile_pad = option.get("tile_pad", 10)
        pre_pad = option.get("pre_pad", 10)
        fp32 = option.get("fp32", False)
        gpu_id = option.get("gpu_id", 0)

        self.output_scale = option.get("output_scale", None)

        model, netscale, file_url = load_model()
        model_name = "realesr-animevideov3"  # x4 VGG-style model (XS size)
        model_path = prepare_model(model_name, file_url)

        self.upsampler = RealESRGANer(
            scale=netscale,
            model_path=model_path,
            dni_weight=None,
            model=model,
            tile=tile,
            tile_pad=tile_pad,
            pre_pad=pre_pad,
            half=not fp32,
            gpu_id=gpu_id,
        )

    def process(self, task):
        output_queue = task.get_outputs().get(0, None)
        input_queue = task.get_inputs().get(0, None)

        while not input_queue.empty():
            pkt = input_queue.get()
            # process EOS
            if pkt.timestamp == Timestamp.EOF:
                Log.log_node(LogLevel.INFO, task.get_node(), "Receive EOF")
                if output_queue is not None:
                    output_queue.put(Packet.generate_eof_packet())
                task.timestamp = Timestamp.DONE
                return ProcessResult.OK

            video_frame = pkt.get(VideoFrame)
            # use ffmpeg
            frame = ffmpeg.reformat(video_frame,
                                    "rgb24").frame().plane(0).numpy()

            output, _ = self.upsampler.enhance(frame, self.output_scale)
            Log.log_node(
                LogLevel.INFO,
                self._node,
                "enhance output shape: ",
                output.shape,
                " flags: ",
                output.flags,
            )
            output = np.ascontiguousarray(output)
            rgbformat = mp.PixelInfo(mp.kPF_RGB24)
            image = mp.Frame(mp.from_numpy(output), rgbformat)

            output_frame = VideoFrame(image)
            Log.log_node(LogLevel.INFO, self._node, "output video frame")

            output_frame.pts = video_frame.pts
            output_frame.time_base = video_frame.time_base
            output_pkt = Packet(output_frame)
            output_pkt.timestamp = pkt.timestamp
            if output_queue is not None:
                output_queue.put(output_pkt)

        return ProcessResult.OK
