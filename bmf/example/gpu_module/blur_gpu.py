import re
from math import pi, cos, tan, sqrt

from bmf import *
import hmp
from cuda import cuda
import cvcuda
import torch

import pdb

class blur_gpu(Module):
    def __get_blur(self, option):
        self.size = option['size'] if 'size' in option else [1, 1]
        self.planes = option['planes'] if 'planes' in option else 0xf
        self.border = option['border'] if 'border' in option else cvcuda.Border.CONSTANT
        size_tensor = torch.tensor(self.size, dtype=torch.int, device='cuda')
        sizet = torch.ones((4, 2), dtype=torch.int, device='cuda') * size_tensor
        if 'op' in option.keys():
            op = option['op']
        else:
            Log.log(LogLevel.ERROR, "Blur op unspecified (op supported: gblur/avgblur/median)")
        if 'gblur' == op:
            self.blur_op = cvcuda.gaussian_into
            self.sigma = option['sigma'] if 'sigma' in option else [0.5, 0.5]
            self.steps = option['steps'] if 'steps' in option else 1
            sigma_tensor = torch.tensor(self.sigma, dtype=torch.double, device='cuda')
            sigmat = torch.ones((4,2), dtype=torch.double, device='cuda') * sigma_tensor
            self.op_args = [self.size, cvcuda.as_tensor(sizet), cvcuda.as_tensor(sigmat), self.border]
        elif 'avgblur' == op:
            self.blur_op = cvcuda.averageblur_into
            self.anchor = option['anchor'] if 'anchor' in option else [-1, -1]
            anchor_tensor = torch.tensor(self.anchor, dtype=torch.int, device='cuda')
            anchort = torch.ones((4,2), dtype=torch.int, device='cuda') * anchor_tensor
            self.op_args = [self.size, cvcuda.as_tensor(sizet), cvcuda.as_tensor(anchort), self.border]
        elif 'median' == op:
            self.blur_op = cvcuda.median_blur_into
            self.percentile = option['percentile'] if 'percentile' in option else 0.5
            self.size = option['radius'] if 'radius' in option else self.size
            size_tensor = torch.tensor(self.size, dtype=torch.int, device='cuda')
            sizet = torch.ones((4, 2), dtype=torch.int, device='cuda') * size_tensor
            self.op_args = [cvcuda.as_tensor(sizet)]
        else:
            Log.log(LogLevel.ERROR, "Unsupported blur operating")

    def __init__(self, node, option=None):
        self.node_ = node
        self.option_ = option
        self.eof_received_ = False

        self.__get_blur(option)

        self.i420info = hmp.PixelInfo(hmp.PixelFormat.kPF_YUV420P, hmp.ColorSpace.kCS_BT470BG, hmp.ColorRange.kCR_MPEG)
        self.i420_out = None

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

            # validate crop parameters
            # if (not 0 <= self.x < in_frame.width - 1) or (not 0 <= self.y < in_frame.height - 1):
            #     Log.log(LogLevel.ERROR, "Invalid crop position")
            # if (self.x + self.width >= in_frame.width):
            #     Log.log(LogLevel.ERROR, "Crop width is out of image bound")
            # if (self.y + self.height >= in_frame.height):
            #     Log.log(LogLevel.ERROR, "Crop height is out of image bound")

            if (in_frame.frame().device() == hmp.Device('cpu')):
                in_frame = in_frame.cuda()
            tensor_list = in_frame.frame().data()
            frame_out = hmp.Frame(in_frame.width, in_frame.height, in_frame.frame().pix_info(), device='cuda')

            out_tensor_list = frame_out.data()
            stream = hmp.current_stream(hmp.kCUDA)
            cvstream = cvcuda.cuda.as_stream(stream.handle())

            in_list = []
            out_list = []
            cvimg_batch = cvcuda.ImageBatchVarShape(4)
            cvimg_batch_out = cvcuda.ImageBatchVarShape(4)

            # deal with nv12 special case
            if (in_frame.frame().format() == hmp.PixelFormat.kPF_NV12):

                self.i420_in = hmp.Frame(in_frame.width, in_frame.height, self.i420info, device='cuda')
                self.i420_out = hmp.Frame(in_frame.width, in_frame.height, self.i420info, device='cuda')
                hmp.img.yuv_to_yuv(self.i420_in.data(), in_frame.frame().data(), self.i420info, in_frame.frame().pix_info())
                in_list = [x.torch() for x in self.i420_in.data()]
                out_list = [x.torch() for x in self.i420_out.data()]
                # cvimg_batch.pushback([cvcuda.as_image(x) for x in in_list])
                # cvimg_batch_out.pushback([cvcuda.as_image(x) for x in out_list])

            # other pixel formats, e.g. yuv420, rgb
            else:
                in_list = [x.torch() for x in tensor_list]
                out_list = [x.torch() for x in out_tensor_list]

            for index, (in_tensor, out_tensor) in enumerate(zip(in_list, out_list)):
                # chroma_shift = 0
                # # rect = cvcuda.RectI(self.x, self.y, self.width, self.height)
                # if ((index in range(1,3)) and
                #     (in_frame.frame().format() == hmp.PixelFormat.kPF_NV12 or
                #      in_frame.frame().format() == hmp.PixelFormat.kPF_YUV420P)):
                #     chroma_shift = 1
                # rect = cvcuda.RectI(self.x >> chroma_shift, self.y >> chroma_shift,
                #                     self.width >> chroma_shift, self.height >> chroma_shift)
                if (((1 << index) & self.planes) != 0):
                    cvimg_batch.pushback(cvcuda.as_image(in_tensor))
                    cvimg_batch_out.pushback(cvcuda.as_image(out_tensor))

            self.blur_op(cvimg_batch_out, cvimg_batch, *self.op_args, stream=cvstream)

            if (in_frame.frame().format() == hmp.PixelFormat.kPF_NV12):
                hmp.img.yuv_to_yuv(frame_out.data(), self.i420_out.data(), frame_out.pix_info(), self.i420_out.pix_info())

            videoframe_out = VideoFrame(frame_out)
            videoframe_out.pts = in_frame.pts
            videoframe_out.time_base = in_frame.time_base
            out_pkt = Packet(videoframe_out)
            out_pkt.timestamp = videoframe_out.pts
            output_queue.put(out_pkt)

        if self.eof_received_:
            output_queue.put(Packet.generate_eof_packet())
            Log.log_node(LogLevel.DEBUG, self.node_,
                         'output stream', 'done')
            task.set_timestamp(Timestamp.DONE)

        return ProcessResult.OK