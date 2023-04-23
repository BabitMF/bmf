import re

from bmf import *
import hmp
from cuda import cuda
import cvcuda
import torch

import pdb

class scale_cvcuda(Module):
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
        self.uv_tensor_out = torch.empty((2, self.h // 2, self.w // 2), dtype=torch.uint8, device='cuda')
        # if 'to_gpu' in option.keys():
        #     self.trans_to_gpu_ = option['to_gpu']
        # self.gpu_alloced_ = []
        # self.init_context_flag_ = False
        # self.ctx = None
    
    def process(self, task):

        # get input and output packet queue
        input_queue = task.get_inputs()[0]
        output_queue = task.get_outputs()[0]

        # if (not self.init_context_flag_):
        #     self.init_context_flag_ = True
        #     err, cuDevice = cuda.cuDeviceGet(0)
        #     err, self.ctx = cuda.cuCtxCreate(0, cuDevice)

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
            torch_list = []
            # pdb.set_trace()
            frame_out = hmp.Frame(self.w, self.h, in_frame.frame().pix_info(), device='cuda')
            # videoframe_out = VideoFrame(self.w, self.h, in_frame.frame().pix_info(), device='cuda')
            # frame_out = videoframe_out.frame()
            out_list = frame_out.data()
            stream = hmp.current_stream(hmp.kCUDA)
            cvstream = cvcuda.cuda.as_stream(stream.handle())
            
            # deal with nv12 special case
            if (in_frame.frame().format() == hmp.PixelFormat.kPF_NV12):
                cvimg_batch = cvcuda.ImageBatchVarShape(3)
                cvimg_batch_out = cvcuda.ImageBatchVarShape(3)

                uv_tensor = tensor_list[1].torch().permute(2,0,1).contiguous()
                ut = uv_tensor[0, :, :]
                vt = uv_tensor[1, :, :]
                yimg = cvcuda.as_image(tensor_list[0].torch())
                uimg = cvcuda.as_image(ut)
                vimg = cvcuda.as_image(vt)
                cvimg_batch.pushback([yimg, uimg, vimg])

                # uv_tensor_out = torch.empty((2, self.h // 2, self.w // 2), dtype=torch.uint8, device='cuda')
                # uv_tensor_out = frame_out.plane(1).torch().permute(2, 1, 0).contiguous()
                yimg_out = cvcuda.as_image(frame_out.plane(0).torch())
                uimg_out = cvcuda.as_image(self.uv_tensor_out[0,:,:])
                vimg_out = cvcuda.as_image(self.uv_tensor_out[1,:,:])
                cvimg_batch_out.pushback([yimg_out, uimg_out, vimg_out])

                cvcuda.resize_into(cvimg_batch_out, cvimg_batch, self.algo, stream=cvstream)

                uv_plane = frame_out.plane(1).torch()
                uv_plane[:] = self.uv_tensor_out.permute(1,2,0)[:]

            # other pixel formats, e.g. yuv420, rgb
            else:
                cvimg_batch = cvcuda.ImageBatchVarShape(in_frame.frame().nplanes())
                cvimg_batch_out = cvcuda.ImageBatchVarShape(in_frame.frame().nplanes())

                for t, f in zip(tensor_list, out_list):
                    cvimg = cvcuda.as_image(t.torch())
                    cvimg_out = cvcuda.as_image(f.torch())
                    cvimg_batch.pushback(cvimg)
                    cvimg_batch_out.pushback(cvimg_out)
                cvcuda.resize_into(cvimg_batch_out, cvimg_batch, self.algo, stream=cvstream)
            
            videoframe_out = VideoFrame(frame_out)
            videoframe_out.pts = in_frame.pts
            videoframe_out.time_base = in_frame.time_base
            out_pkt = Packet(videoframe_out.cpu())
            out_pkt.timestamp = videoframe_out.pts
            output_queue.put(out_pkt)

        if self.eof_received_:
            output_queue.put(Packet.generate_eof_packet())
            Log.log_node(LogLevel.DEBUG, self.node_,
                         'output stream', 'done')
            task.set_timestamp(Timestamp.DONE)
            # if self.ctx is not None:
            #     self.ctx.pop()
        return ProcessResult.OK
