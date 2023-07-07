import tensorrt as trt
import torch
import torch.nn.functional as F
import numpy as np
import sys
import time

if sys.version_info.major == 2:
    from Queue import Queue
else:
    from queue import Queue

import PIL
from PIL import Image

sys.path.append("../../")

from bmf import *
import bmf.hml.hmp as mp

class DetectResult(object):
    def __init__(self,x1,y1,x2,y2,label,score,pts=0):
        self.x1_ = x1
        self.y1_ = y1
        self.x2_ = x2
        self.y2_ = y2
        self.label_ = label
        self.score_ = score

    def __str__(self):
        msg="x1:{},y1:{},x2:{},y2:{},label:{},score:{}".format(self.x1_,self.y1_,self.x2_,self.y2_,self.label_,self.score_)
        return msg

    def __repr__(self):
        msg = "x1:{},y1:{},x2:{},y2:{},label:{},score:{}".format(self.x1_, self.y1_, self.x2_, self.y2_, self.label_,
                                                                 self.score_)
        return msg

class trt_face_detect(Module):
    def __init__(self, node=None, option=None):
        self.node_ = node
        self.option_ = option

        if option is None:
            Log.log(LogLevel.ERROR, "Option is none")
            return

        if "model_path" in option.keys():
            self.model_path_ = option["model_path"]

        if "label_to_frame" in option.keys():
            self.label_frame_flag_ = option["label_to_frame"]

        if "input_shapes" in option.keys():
            self.input_shapes_ = option["input_shapes"]

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
        self.in_frame_num_ = 1
        self.out_frame_num_ = 1

        self.eof_received_ = False

        Log.log(LogLevel.ERROR, "Load model takes", (time.time() - start_time))

    def reset(self):
        self.eof_received_ = False
        while not self.frame_cache_.empty():
            self.frame_cache_.get()

    def to_scalar_types(self, trt_dtype):
        dtype_map = {
            trt.float32: mp.kFloat32,
            trt.float16: mp.kHalf,
            trt.int32: mp.kInt32,
            trt.int8: mp.kInt8,
            trt.uint8: mp.kUInt8,
        }
        return dtype_map[trt_dtype]

    def pre_process(self, torch_image_array):
        input_shape = list(self.input_shapes_.values())[0]
        # input shape is the shape of trt engine
        batch = input_shape[0]
        channel = input_shape[1]
        width = input_shape[3]
        height = input_shape[2]

        input_tensor = torch.stack(torch_image_array).float()
        input_tensor = torch.permute(input_tensor, [0, 3, 1, 2])
        input_tensor = F.interpolate(input_tensor, size=(height, width), mode='bilinear')

        torch_mean = torch.empty((1, 3, 1, 1), device="cuda").fill_(0.5)
        torch_std = torch.empty((1, 3, 1, 1), device="cuda").fill_(0.5)

        input_tensor = (input_tensor / 255 - torch_mean) / torch_std

        return input_tensor

    def post_process(self, pil_image_array, boxes, scores):
        output_list = []
        boxes_num = boxes.shape[1]
        for image_id in range(len(pil_image_array)):
            image = pil_image_array[image_id]
            box_data = []
            for index in range(boxes_num):
                if (scores[image_id, index, 1] > 0.8):
                    box = boxes[image_id, index, :]
                    x1 = int(box[0] * image.size[0])
                    y1 = int(box[1] * image.size[1])
                    x2 = int(box[2] * image.size[0])
                    y2 = int(box[3] * image.size[1])
                    detect_result = DetectResult(x1, y1, x2, y2, 1, scores[image_id, index, 1])
                    box_data.append(detect_result)
            output_list.append(box_data)
        return output_list

    def label_frame(self, input_frames, pil_image_array, detect_result_list):
        from PIL import ImageDraw
        output_frame_list = []
        for index_frame in range(len(pil_image_array)):
            image = pil_image_array[index_frame]
            draw = ImageDraw.Draw(image)
            for index_box in range(len(detect_result_list[index_frame])):
                detect_result = detect_result_list[index_frame][index_box]
                draw.rectangle([detect_result.x1_, detect_result.y1_, detect_result.x2_, detect_result.y2_])
            del draw
            numpy_image = np.asarray(image)
            H420 = mp.PixelInfo(mp.kPF_YUV420P)
            rgb = mp.PixelInfo(mp.kPF_RGB24)

            frame = mp.Frame(mp.from_numpy(np.ascontiguousarray(numpy_image)), rgb) 
            out_frame = VideoFrame(frame).reformat(H420)

            out_frame.pts = input_frames[index_frame].pts
            out_frame.time_base = input_frames[index_frame].time_base
            output_frame_list.append(out_frame)
        return output_frame_list

    def inference(self):
        frame_num = min(self.frame_cache_.qsize(), self.in_frame_num_)
        input_frames = []
        
        if frame_num == 0:
            return [], []
        torch_image_array = []
        pil_image_array = []
        for i in range(frame_num):
            vf = self.frame_cache_.get()
            if (vf.frame().device() == mp.Device('cpu')):
                vf = vf.cuda()
            input_frames.append(vf)

            rgb = mp.PixelInfo(mp.kPF_RGB24)
            torch_vf = vf.reformat(rgb).frame().plane(0).torch()
            numpy_vf = torch_vf.cpu().numpy()
            torch_image_array.append(torch_vf)
            pil_image_array.append(PIL.Image.fromarray(numpy_vf))

        input_tensor = self.pre_process(torch_image_array)

        for i in range(self.num_inputs_):
            self.context_.set_tensor_address(self.tensor_names_[i], int(input_tensor.contiguous().data_ptr()))

        for i in range(self.num_inputs_, self.num_io_tensors_):
            self.context_.set_tensor_address(self.tensor_names_[i], int(self.output_dict_[self.tensor_names_[i]].data_ptr()))

        self.context_.execute_async_v3(self.stream_.handle())

        scores = self.output_dict_["scores"].cpu().numpy()
        boxes = self.output_dict_["boxes"].cpu().numpy()

        detect_result_list = self.post_process(pil_image_array, boxes, scores)
        if self.label_frame_flag_ == 1:
            result_frames = self.label_frame(input_frames, pil_image_array, detect_result_list)
            return result_frames, detect_result_list

        return input_frames, detect_result_list
    
    def process(self, task):
        input_queue = task.get_inputs()[0]
        output_queue_0 = task.get_outputs()[0]
        output_queue_size = len(task.get_outputs())
        if output_queue_size >= 2:
            output_queue_1 = task.get_outputs()[1]

        while not input_queue.empty():
            pkt = input_queue.get()
            if pkt.timestamp == Timestamp.EOF:
                self.eof_received_ = True
            if pkt.is_(VideoFrame):
                self.frame_cache_.put(pkt.get(VideoFrame))

        while self.frame_cache_.qsize() >= self.in_frame_num_ or self.eof_received_:
            out_frames, detect_result_list = self.inference()
            for idx, frame in enumerate(out_frames):
                pkt = Packet(frame)
                pkt.timestamp = frame.pts
                output_queue_0.put(pkt)

                if (output_queue_size >= 2):
                    pkt = Packet(detect_result_list[idx])
                    pkt.timestamp = frame.pts
                    output_queue_1.put(pkt)

            if self.frame_cache_.empty():
                break

        if self.eof_received_:
            for key in task.get_outputs():
                task.get_outputs()[key].put(Packet.generate_eof_packet())
                Log.log_node(LogLevel.DEBUG, self.node_, "output stream", "done")
            task.timestamp = Timestamp.DONE

        return ProcessResult.OK
