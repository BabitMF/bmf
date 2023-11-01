import re

from bmf import *
import bmf.hml.hmp as hmp
import cvcuda


class scale_gpu(Module):

    def __get_size(self, size_str):
        split = re.split('\*|x|,', size_str)
        w = int(split[0])
        h = int(split[1])
        return w, h

    def __get_algo(self, algo_str):
        return {
            'area': cvcuda.Interp.AREA,
            'cubic': cvcuda.Interp.CUBIC,
            'linear': cvcuda.Interp.LINEAR,
            'nearest': cvcuda.Interp.NEAREST
        }.get(algo_str, cvcuda.Interp.LINEAR)

    def __init__(self, node, option=None):
        self.node_ = node
        self.option_ = option
        self.eof_received_ = False
        self.algo = cvcuda.Interp.LINEAR
        if 'size' in option.keys():
            self.w, self.h = self.__get_size(option['size'])
        else:
            Log.log(LogLevel.ERROR, "No output size (WxH) provided")
        if 'algo' in option.keys():
            self.algo = self.__get_algo(option['algo'])

        # self.uv_tensor_out = numpy.empty((2, self.h // 2, self.w // 2), dtype='uint8')
        self.i420info = hmp.PixelInfo(hmp.PixelFormat.kPF_YUV420P,
                                      hmp.ColorSpace.kCS_BT470BG,
                                      hmp.ColorRange.kCR_MPEG)
        self.u420info = hmp.PixelInfo(hmp.PixelFormat.kPF_YUV420P10LE,
                                      hmp.ColorSpace.kCS_BT2020_CL,
                                      hmp.ColorRange.kCR_MPEG)
        self.i420_out = hmp.Frame(self.w, self.h, self.i420info, device='cuda')
        self.u420_out = hmp.Frame(self.w, self.h, self.u420info, device='cuda')
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
            # pdb.set_trace()
            if (in_frame.frame().device() == hmp.Device('cpu')):
                in_frame = in_frame.cuda()
            tensor_list = in_frame.frame().data()
            # frame_out = hmp.Frame(self.w, self.h, in_frame.frame().pix_info(), device='cuda')
            videoframe_out = VideoFrame(self.w,
                                        self.h,
                                        in_frame.frame().pix_info(),
                                        device='cuda')
            frame_out = videoframe_out.frame()

            out_list = frame_out.data()
            stream = hmp.current_stream(hmp.kCUDA)
            cvstream = cvcuda.cuda.as_stream(stream.handle())

            # deal with nv12 special case
            if (in_frame.frame().format() == hmp.PixelFormat.kPF_NV12 or
                    in_frame.frame().format() == hmp.PixelFormat.kPF_P010LE):
                cvimg_batch = cvcuda.ImageBatchVarShape(3)
                cvimg_batch_out = cvcuda.ImageBatchVarShape(3)

                in_420 = hmp.Frame(in_frame.width,
                                   in_frame.height,
                                   self.pinfo_map[in_frame.frame().format()],
                                   device='cuda')
                out_420 = hmp.Frame(self.w,
                                    self.h,
                                    self.pinfo_map[in_frame.frame().format()],
                                    device='cuda')
                hmp.img.yuv_to_yuv(in_420.data(),
                                   in_frame.frame().data(),
                                   self.pinfo_map[in_frame.frame().format()],
                                   in_frame.frame().pix_info())
                in_list = in_420.data()
                out_list = out_420.data()
                cvimg_batch.pushback([cvcuda.as_image(x) for x in in_list])
                cvimg_batch_out.pushback(
                    [cvcuda.as_image(x) for x in out_list])

                cvcuda.resize_into(cvimg_batch_out,
                                   cvimg_batch,
                                   self.algo,
                                   stream=cvstream)

                hmp.img.yuv_to_yuv(frame_out.data(), out_420.data(),
                                   frame_out.pix_info(), out_420.pix_info())

            # other pixel formats, e.g. yuv420, rgb
            else:
                cvimg_batch = cvcuda.ImageBatchVarShape(
                    in_frame.frame().nplanes())
                cvimg_batch_out = cvcuda.ImageBatchVarShape(
                    in_frame.frame().nplanes())

                for t, f in zip(tensor_list, out_list):
                    cvimg = cvcuda.as_image(t)
                    cvimg_out = cvcuda.as_image(f)
                    cvimg_batch.pushback(cvimg)
                    cvimg_batch_out.pushback(cvimg_out)

                cvcuda.resize_into(cvimg_batch_out,
                                   cvimg_batch,
                                   self.algo,
                                   stream=cvstream)

            # videoframe_out = VideoFrame(frame_out)
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


def register_scale_gpu_info(info):
    info.module_description = "Builtin module for scale using GPU"
    info.module_tag = ModuleTag.TAG_DEVICE_HWACCEL
