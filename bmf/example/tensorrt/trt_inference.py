from bmf import *
import hmp as mp
from cuda import cudart
import tensorrt as trt
import numpy as np
import torch

if sys.version_info.major == 2:
    from Queue import Queue
else:
    from queue import Queue

def to_scalar_types(trt_dtype):
    dtype_map = {
        trt.float32: mp.kFloat32,
        trt.float16: mp.kHalf,
        trt.int32: mp.kInt32,
        trt.int8: mp.kInt8,
    }
    return dtype_map[trt_dtype]
     

class trt_inference(Module):
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

        if 'pre_process' in option.keys():
            self.pre_process_ = option['pre_process']
        else:
            self.pre_process_ = self.default_pre_process
        
        if 'post_process' in option.keys():
            self.post_process_ = option['post_process']
        else:
            self.post_process_ = self.default_post_process
        
        if 'batch_size' in option.keys():
            self.batch_size_ = option['batch_size']
        
        if 'in_frame_num' in option.keys():
            self.in_frame_num_ = option['in_frame_num']
        else:
            Log.log(LogLevel.INFO, "No in_frame_num keyword, set it 1 instead.")
            self.in_frame_num_ = 1
        
        if 'out_frame_num' in option.keys():
            self.out_frame_num_ = option['out_frame_num']
        else:
            Log.log(LogLevel.INFO, "No out_frame_num keyword, set it 1 instead.")
            self.out_frame_num_ = 1
        
        logger = trt.Logger(trt.Logger.ERROR)
        self.engine_ = trt.Runtime(logger).deserialize_cuda_engine(self.model_path_)

        if self.engine_ is None:
            Log.log(LogLevel.ERROR, "Failed building engine!")
            return
        Log.log(LogLevel.INFO, "Succeeded building engine!")

        self.num_io_tensors_ = self.engine_.num_io_tensors
        self.tensor_names_ = [self.engine_.get_tensor_name(i) for i in range(self.num_io_tensors_)]
        self.num_inputs_ = [self.engine_.get_tensor_mode(self.tensor_names_[i]) for i in range(self.num_io_tensors_)] \
                           .count(trt.TensorIOMode.INPUT)
        self.num_outputs_ = [self.engine_.get_tensor_mode(self.tensor_names_[i]) for i in range(self.num_io_tensors_)] \
                           .count(trt.TensorIOMode.OUTPUT)

        self.context_ = self.engine_.create_execution_context()
        self.stream_ = mp.current_stream(mp.kCUDA)

        for i in range(self.num_inputs_):
            self.context_.set_input_shape(self.tensor_names_[0], self.input_shapes_[self.tensor_names_[0]])
        
        self.output_dict_ = dict()
        for i in range(self.num_inputs_, self.num_io_tensors_):
            self.output_dict_[self.tensor_names_[i]] = mp.empty(self.context_.get_tensor_shape[self.tensor_names_[i]],
                                                                device=mp.kCUDA,
                                                                dtype=to_scalar_types(self.engine_.get_tensor_dtype(self.tensor_names_[i])))

        self.frame_cache_ = Queue()

        self.eof_received_ = False
    
    # default_pre_process just combines batch_size frames into one input tensor for tensorrt
    def default_pre_process(self, frame_cache, in_frame_num):
        assert len(self.num_inputs_) == 1, "default_pre_process can only be applied on the model with single input, \
                                            write a new customized process for your model."
        frame_num = min(frame_cache, in_frame_num)
        input_frames = []
        input_torch_arrays = []
        for i in range(frame_num):
            vf = frame_cache.queue[i]
            if (vf.frame().device() == mp.Device('cpu')):
                vf = vf.cuda()
            input_frames.append(vf)
            vf_image = vf.to_image(mp.kNHWC)
            input_torch_arrays.append(vf_image.torch())
        # for the last few frames, repeat the last frame
        for i in range(in_frame_num - frame_num):
            input_torch_arrays.append(input_torch_arrays[-1])
        
        input_dict = dict()
        input_dict[self.tensor_names_[0]] = torch.stack(input_torch_arrays, 0).data_ptr()

        return input_dict
    
    # default_post_process just separate batched tensor into multiple frames
    def default_post_process(self, output_dict, out_frame_num):
        assert len(self.num_outputs_) == 1, "default_post_process can only be applied on the model with single input, \
                                             write a new customized process for your model."
        output_tensor = output_dict[self.tensor_names_[1]]

        output_tensor_torch = output_tensor.torch()
        out_frames = []

        for i in range(out_frame_num):
            H420 = mp.PixelInfo(mp.kPF_YUV420P)
            image = mp.Image(mp.from_torch(output_tensor_torch[i], format=mp.kNHWC))
            out_frame = VideoFrame(image).to_frame(H420)

            if self.frame_cache_.empty():
                break

            input_frame = self.frame_cache_.get()
            out_frame.pts = input_frame.pts
            out_frame.time_base = input_frame.time_base
            out_frames.append(out_frame)
        
        return out_frames

    def inference(self):
        input_dict = self.pre_process_(self.frame_cache_, self.batch_size_)

        for i in range(self.num_inputs):
            self.context_.set_tensor_address(self.tensor_names_[i], int(input_dict[self.tensor_names_[i]]))
        
        for i in range(self.num_inputs_, self.num_io_tensors_):
            self.context_.set_tensor_address(self.tensor_names_[i], int(self.output_dict_[self.tensor_names[i]].torch().data_ptr()))
        
        self.context_.execute_async_v3(self.stream_.handle())

        return self.post_process_(self.output_dict_, self.out_frame_num_)

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
