from bmf import *
import hmp
from cuda import cuda
import cvcuda
import torch

import pdb


class gamma_gpu(Module):

    def __get_gamma(self, option):
        gamma = [1.0]
        # user-provided gamma value should have the same channel number as the image
        # e.g. [0.1, 0.2, 0.3, 0.4] for rgba images
        if 'gamma' in option.keys():
            gamma = option['gamma']
            self.gamma = cvcuda.as_tensor(
                torch.tensor(gamma, device='cuda', dtype=torch.float))

    def __init__(self, node, option=None):
        self.node_ = node
        self.option_ = option
        self.eof_received_ = False

        self.__get_gamma(option)

        self.i420info = hmp.PixelInfo(hmp.PixelFormat.kPF_YUV420P,
                                      hmp.ColorSpace.kCS_BT470BG,
                                      hmp.ColorRange.kCR_MPEG)
        self.i420_out = None
        self.fmt_list = [
            hmp.PixelFormat.kPF_BGR24, hmp.PixelFormat.kPF_BGRA32,
            hmp.PixelFormat.kPF_RGB24, hmp.PixelFormat.kPF_RGBA32,
            hmp.PixelFormat.kPF_GRAY8
        ]

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
            frame_out = hmp.Frame(in_frame.width,
                                  in_frame.height,
                                  in_frame.frame().pix_info(),
                                  device='cuda')

            out_tensor_list = frame_out.data()
            stream = hmp.current_stream(hmp.kCUDA)
            cvstream = cvcuda.cuda.as_stream(stream.handle())

            in_list = []
            out_list = []
            cvimg_batch = cvcuda.ImageBatchVarShape(1)
            cvimg_batch_out = cvcuda.ImageBatchVarShape(1)

            if in_frame.frame().format() in self.fmt_list:
                in_img = cvcuda.as_image(in_frame.frame().plane(0).torch())
                out_img = cvcuda.as_image(frame_out.plane(0).torch())
                cvimg_batch.pushback(in_img)
                cvimg_batch_out.pushback(out_img)

                cvcuda.gamma_contrast_into(cvimg_batch_out,
                                           cvimg_batch,
                                           self.gamma,
                                           stream=cvstream)
            else:
                Log.log(
                    LogLevel.ERROR,
                    "Unsupported pixel format. Gamma module only supports rgb/rgba formats"
                )
                sys.exit(1)
            '''
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
            '''
            videoframe_out = VideoFrame(frame_out)
            videoframe_out.pts = in_frame.pts
            videoframe_out.time_base = in_frame.time_base
            out_pkt = Packet(videoframe_out.cpu())
            out_pkt.timestamp = videoframe_out.pts
            output_queue.put(out_pkt)

        if self.eof_received_:
            output_queue.put(Packet.generate_eof_packet())
            Log.log_node(LogLevel.DEBUG, self.node_, 'output stream', 'done')
            task.set_timestamp(Timestamp.DONE)

        return ProcessResult.OK


def register_gamma_gpu_info(info):
    info.module_description = "Builtin module for gamma using GPU"
    info.module_tag = ModuleTag.TAG_DEVICE_HWACCEL
