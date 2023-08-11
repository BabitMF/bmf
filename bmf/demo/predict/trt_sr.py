import numpy as np
import torch
import sys
import time
import tensorrt as trt
from cuda import cudart

sys.path.append("../../")
from bmf import *
import bmf.hml.hmp as mp

if sys.version_info.major == 2:
    from Queue import Queue
else:
    from queue import Queue


class trt_sr(Module):

    def __init__(self, node=None, option=None):
        self.node_ = node
        self.option_ = option

        if option is None:
            Log.log(LogLevel.ERROR, "Option is none")
            return

        if 'model_path' in option.keys():
            self.model_path_ = option['model_path']

        if 'input_shapes' in option.keys():
            self.input_shapes_ = option['input_shapes']

        start_time = time.time()

        logger = trt.Logger(trt.Logger.ERROR)
        with open(self.model_path_, 'rb') as f:
            engine_buffer = f.read()
        self.engine_ = trt.Runtime(logger).deserialize_cuda_engine(
            engine_buffer)

        if self.engine_ is None:
            Log.log(LogLevel.ERROR, "Failed building engine!")
            return
        Log.log(LogLevel.INFO, "Succeeded building engine!")

        self.num_io_tensors_ = self.engine_.num_io_tensors
        self.tensor_names_ = [
            self.engine_.get_tensor_name(i)
            for i in range(self.num_io_tensors_)
        ]
        self.num_inputs_ = [self.engine_.get_tensor_mode(self.tensor_names_[i]) for i in range(self.num_io_tensors_)] \
                           .count(trt.TensorIOMode.INPUT)
        assert self.num_inputs_ == len(
            self.input_shapes_.keys()
        ), "The number of input_shapes doesn't match the number of model's inputs."
        self.num_outputs_ = [self.engine_.get_tensor_mode(self.tensor_names_[i]) for i in range(self.num_io_tensors_)] \
                           .count(trt.TensorIOMode.OUTPUT)

        self.context_ = self.engine_.create_execution_context()
        self.stream_ = mp.current_stream(mp.kCUDA)

        for i in range(self.num_inputs_):
            self.context_.set_input_shape(
                self.tensor_names_[0],
                self.input_shapes_[self.tensor_names_[0]])

        self.output_dict_ = dict()
        for i in range(self.num_inputs_, self.num_io_tensors_):
            self.output_dict_[self.tensor_names_[i]] = mp.empty(
                self.context_.get_tensor_shape(self.tensor_names_[i]),
                device=mp.kCUDA,
                dtype=self.to_scalar_types(
                    self.engine_.get_tensor_dtype(self.tensor_names_[i])))

        self.frame_cache_ = Queue()

        self.in_frame_num_ = 7
        self.out_frame_num_ = 3

        self.eof_received_ = False

        Log.log(LogLevel.ERROR, "Load model takes", (time.time() - start_time))

    def to_scalar_types(self, trt_dtype):
        dtype_map = {
            trt.float32: mp.kFloat32,
            trt.float16: mp.kHalf,
            trt.int32: mp.kInt32,
            trt.int8: mp.kInt8,
            trt.uint8: mp.kUInt8,
        }
        return dtype_map[trt_dtype]

    def reset(self):
        # clear status
        self.eof_received_ = False
        while not self.frame_cache_.empty():
            self.frame_cache_.get()

    def inference(self):
        frame_num = min(self.frame_cache_.qsize(), self.in_frame_num_)
        input_frames = []
        input_torch_array = []
        for i in range(frame_num):
            vf = self.frame_cache_.queue[i]
            if (vf.frame().device() == mp.Device('cpu')):
                vf = vf.cuda()
            input_frames.append(vf)

            rgb = mp.PixelInfo(mp.kPF_RGB24)
            torch_vf = torch.from_dlpack(vf.reformat(rgb).frame().plane(0))
            input_torch_array.append(torch_vf)
        for i in range(self.in_frame_num_, frame_num):
            input_torch_array.append(input_torch_array[-1])

        input_tensor = torch.concat(input_torch_array, 2)

        for i in range(self.num_inputs_):
            self.context_.set_tensor_address(self.tensor_names_[i],
                                             int(input_tensor.data_ptr()))

        for i in range(self.num_inputs_, self.num_io_tensors_):
            self.context_.set_tensor_address(
                self.tensor_names_[i],
                int(self.output_dict_[self.tensor_names_[i]].data_ptr()))

        self.context_.execute_async_v3(self.stream_.handle())

        output_tensor = torch.from_dlpack(
            self.output_dict_[self.tensor_names_[-1]])
        output_tensor = torch.squeeze(output_tensor)
        output_tensor = torch.split(output_tensor, self.out_frame_num_, dim=2)

        out_vframes = []

        for i in range(self.out_frame_num_):
            NV12 = mp.PixelInfo(mp.PixelFormat.kPF_NV12,
                                mp.ColorSpace.kCS_BT470BG,
                                mp.ColorRange.kCR_MPEG)
            RGB = mp.PixelInfo(mp.PixelFormat.kPF_RGB24,
                               mp.ColorSpace.kCS_BT709, mp.ColorRange.kCR_MPEG)
            frame = mp.Frame(mp.from_dlpack(output_tensor[i].contiguous()), RGB)
            out_frame = mp.Frame(frame.width(),
                                 frame.height(),
                                 NV12,
                                 device='cuda')
            mp.img.rgb_to_yuv(out_frame.data(), frame.plane(0), NV12, mp.kNHWC)

            if self.frame_cache_.empty():
                break

            out_vframe = VideoFrame(out_frame)
            input_frame = self.frame_cache_.get()
            out_vframe.pts = input_frame.pts
            out_vframe.time_base = input_frame.time_base

            out_vframes.append(out_vframe)

        return out_vframes

    def process(self, task):
        # get input and output packet queue
        input_queue = task.get_inputs()[0]
        output_queue = task.get_outputs()[0]

        # add all input frames into frame cache
        while not input_queue.empty():
            pkt = input_queue.get()
            if pkt.timestamp == Timestamp.EOF:
                self.eof_received_ = True
            elif pkt.is_(VideoFrame):
                self.frame_cache_.put(pkt.get(VideoFrame))

        while self.frame_cache_.qsize() >= self.in_frame_num_ or \
                self.eof_received_:
            if self.frame_cache_.qsize() > 0:
                out_frames = self.inference()
                for frame in out_frames:
                    pkt = Packet(frame)
                    pkt.timestamp = frame.pts
                    output_queue.put(pkt)

            if self.frame_cache_.empty():
                break

        # add eof packet to output
        if self.eof_received_:
            output_queue.put(Packet.generate_eof_packet())
            Log.log_node(LogLevel.DEBUG, self.node_, 'output stream', 'done')
            task.timestamp = Timestamp.DONE

        return ProcessResult.OK
