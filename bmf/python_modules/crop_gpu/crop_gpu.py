import re
from math import pi, cos, tan, sqrt
import numpy

from bmf import *
import bmf.hml.hmp as hmp
import cvcuda


class crop_gpu(Module):

    def __get_crop(self, option):
        self.width = None
        self.height = None
        self.x = None
        self.y = None
        if 'width' in option.keys():
            self.width = option['width']
        if 'height' in option.keys():
            self.height = option['height']
        if 'x' in option.keys():
            self.x = option['x']
        if 'y' in option.keys():
            self.y = option['y']

    def __init__(self, node, option=None):
        self.node_ = node
        self.option_ = option
        self.eof_received_ = False

        self.__get_crop(option)

        self.i420info = hmp.PixelInfo(hmp.PixelFormat.kPF_YUV420P,
                                      hmp.ColorSpace.kCS_BT470BG,
                                      hmp.ColorRange.kCR_MPEG)
        self.u420info = hmp.PixelInfo(hmp.PixelFormat.kPF_YUV420P10LE,
                                      hmp.ColorSpace.kCS_BT2020_CL,
                                      hmp.ColorRange.kCR_MPEG)
        self.i420_out = None
        self.pinfo_map = {
            hmp.PixelFormat.kPF_NV12: self.i420info,
            hmp.PixelFormat.kPF_P010LE: self.u420info
        }

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
            if (self.x is None):
                self.x = in_frame.width // 2 - self.width // 2
                self.y = in_frame.height // 2 - self.height // 2

            # validate crop parameters
            if (not 0 <= self.x < in_frame.width - 1) or (
                    not 0 <= self.y < in_frame.height - 1):
                Log.log(LogLevel.ERROR, "Invalid crop position")
            if (self.x + self.width >= in_frame.width):
                Log.log(LogLevel.ERROR, "Crop width is out of image bound")
            if (self.y + self.height >= in_frame.height):
                Log.log(LogLevel.ERROR, "Crop height is out of image bound")

            if (in_frame.frame().device() == hmp.Device('cpu')):
                in_frame = in_frame.cuda()
            tensor_list = in_frame.frame().data()
            frame_out = hmp.Frame(self.width,
                                  self.height,
                                  in_frame.frame().pix_info(),
                                  device='cuda')
            self.i420_out = hmp.Frame(self.width,
                                      self.height,
                                      self.i420info,
                                      device='cuda')

            out_tensor_list = frame_out.data()
            stream = hmp.current_stream(hmp.kCUDA)
            cvstream = cvcuda.cuda.as_stream(stream.handle())

            in_list = []
            out_list = []

            # deal with nv12 special case
            if (in_frame.frame().format() == hmp.PixelFormat.kPF_NV12):
                # in_420 = hmp.Frame(in_frame.width, in_frame.height, self.i420info, device='cuda')
                pinfo = self.pinfo_map[in_frame.frame().format()]
                in_420 = hmp.Frame(in_frame.width,
                                   in_frame.height,
                                   pinfo,
                                   device='cuda')
                out_420 = hmp.Frame(self.width,
                                    self.height,
                                    pinfo,
                                    device='cuda')
                hmp.img.yuv_to_yuv(in_420.data(),
                                   in_frame.frame().data(), pinfo,
                                   in_frame.frame().pix_info())

                in_list = in_420.data()
                out_list = out_420.data()

            # other pixel formats, e.g. yuv420, rgb
            else:
                in_list = tensor_list
                out_list = out_tensor_list

            for index, (in_tensor,
                        out_tensor) in enumerate(zip(in_list, out_list)):
                chroma_shift = 0
                # rect = cvcuda.RectI(self.x, self.y, self.width, self.height)
                if ((index in range(1, 3)) and
                    (in_frame.frame().format() == hmp.PixelFormat.kPF_NV12
                     or in_frame.frame().format()
                     == hmp.PixelFormat.kPF_YUV420P)):
                    chroma_shift = 1
                rect = cvcuda.RectI(self.x >> chroma_shift,
                                    self.y >> chroma_shift,
                                    self.width >> chroma_shift,
                                    self.height >> chroma_shift)

                cvcuda.customcrop_into(cvcuda.as_tensor(out_tensor, 'HWC'),
                                       cvcuda.as_tensor(in_tensor, 'HWC'),
                                       rect=rect,
                                       stream=cvstream)

            if (in_frame.frame().format() == hmp.PixelFormat.kPF_NV12):
                hmp.img.yuv_to_yuv(frame_out.data(), out_420.data(),
                                   frame_out.pix_info(), out_420.pix_info())

            videoframe_out = VideoFrame(frame_out)
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


def register_crop_gpu_info(info):
    info.module_description = "Builtin module for crop using GPU"
    info.module_tag = ModuleTag.TAG_DEVICE_HWACCEL
