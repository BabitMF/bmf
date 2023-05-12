import numpy as np
import torch
import sys
import time
import tensorrt as trt
from cuda import cudart

sys.path.append("../../")
sys.path.append("../tensorrt/")
from bmf import *
import hmp as mp

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
        self.engine_ = trt.Runtime(logger).deserialize_cuda_engine(engine_buffer)
        
        if self.engine_ is None:
            Log.log(LogLevel.ERROR, "Failed building engine!")
            return
        Log.log(LogLevel.INFO, "Succeeded building engine!")

        self.num_io_tensors_ = self.engine_.num_io_tensors
        self.tensor_names_ = [self.engine_.get_tensor_name(i) for i in range(self.num_io_tensors_)]
        self.num_inputs_ = [self.engine_.get_tensor_mode(self.tensor_names_[i]) for i in range(self.num_io_tensors_)] \
                           .count(trt.TensorIOMode.INPUT)
        assert self.num_inputs_ == len(self.input_shapes_.keys()), "The number of input_shapes doesn't match the number of model's inputs."
        self.num_outputs_ = [self.engine_.get_tensor_mode(self.tensor_names_[i]) for i in range(self.num_io_tensors_)] \
                           .count(trt.TensorIOMode.OUTPUT)

        self.context_ = self.engine_.create_execution_context()
        self.stream_ = mp.current_stream(mp.kCUDA)

        for i in range(self.num_inputs_):
            self.context_.set_input_shape(self.tensor_names_[0], self.input_shapes_[self.tensor_names_[0]])
        
        self.output_dict_ = dict()
        for i in range(self.num_inputs_, self.num_io_tensors_):
            self.output_dict_[self.tensor_names_[i]] = mp.empty(self.context_.get_tensor_shape(self.tensor_names_[i]),
                                                                device=mp.kCUDA,
                                                                dtype=self.to_scalar_types(self.engine_.get_tensor_dtype(self.tensor_names_[i])))
        
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
            torch_vf = vf.reformat(rgb).frame().plane(0).torch()
            input_torch_array.append(torch_vf)
        for i in range(self.in_frame_num_, frame_num):
            input_torch_array.append(input_torch_array[-1])

        input_tensor = torch.concat(input_torch_array, 2)

        for i in range(self.num_inputs_):
            self.context_.set_tensor_address(self.tensor_names_[i], int(input_tensor.data_ptr()))

        for i in range(self.num_inputs_, self.num_io_tensors_):
            self.context_.set_tensor_address(self.tensor_names_[i], int(self.output_dict_[self.tensor_names_[i]].torch().data_ptr()))

        self.context_.execute_async_v3(self.stream_.handle())

        output_tensor = self.output_dict_[self.tensor_names_[-1]].torch()
        output_tensor = torch.squeeze(output_tensor)
        output_tensor = torch.split(output_tensor, self.out_frame_num_, dim=2)

        out_frames = []

        for i in range(self.out_frame_num_):
            H420 = mp.PixelInfo(mp.kPF_YUV420P)

            rgb = mp.PixelInfo(mp.kPF_RGB24)
            frame = mp.Frame(mp.from_torch(output_tensor[i].contiguous()), rgb)
            out_frame = VideoFrame(frame).reformat(H420)
            # TODO remove it when the nvenc bug fixed
            out_frame = out_frame.cpu() # for hw encode

            if self.frame_cache_.empty():
                break

            input_frame = self.frame_cache_.get()
            out_frame.pts = input_frame.pts
            out_frame.time_base = input_frame.time_base

            out_frames.append(out_frame)

        return out_frames

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

