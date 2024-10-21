import openvino.runtime as ov
import numpy as np
import bmf.hmp as mp
import time
import sys
if sys.version_info.major == 2:
    from Queue import Queue
else:
    from queue import Queue

from PIL import Image
from bmf import Module, Log, LogLevel, InputType, ProcessResult, Packet, Timestamp, scale_av_pts, av_time_base, \
    BmfCallBackType, VideoFrame, AudioFrame
from nms import NMS
import threading


class openvino_face_detect(Module):

    def __init__(self, node=None, option=None):
        self.node_ = node
        self.option_ = option

        if option is None:
            Log.log(LogLevel.ERROR, "Option is none")
            return

        # model path
        if 'model_path' in option.keys():
            self.model_path_ = option['model_path']

        if 'label_to_frame' in option.keys():
            self.label_frame_flag_ = option['label_to_frame']
        
        if 'threads' in option.keys():
            self.ov_threads_ = option['threads']
        
        start_time = time.time()

        # load model
        core = ov.Core()
        config = {'PERFORMANCE_HINT': 'LATENCY', 'INFERENCE_NUM_THREADS': self.ov_threads_}
        compiled_model = core.compile_model(self.model_path_, "CPU", config)
        self.infer_request = compiled_model.create_infer_request()

        self.input_shapes_ = []
        for input in compiled_model.inputs:
            self.input_shapes_.append(list(input.shape))

        # batch frame cache
        self.frame_cache_ = Queue()

        self.in_frame_num_ = 1
        self.out_frame_num_ = 1

        self.eof_received_ = False

        Log.log(LogLevel.ERROR, "Load model takes", (time.time() - start_time))

    def reset(self):
        # clear status
        self.eof_received_ = False
        while not self.frame_cache_.empty():
            self.frame_cache_.get()

    def pre_process(self, image_list):

        image_size = (self.input_shapes_[0][3], self.input_shapes_[0][2])
        pre_result = None
        for image in image_list:
            img = image.resize(image_size, Image.BILINEAR)
            img_data = np.array(img)
            img_data = np.transpose(img_data, [2, 0, 1])
            img_data = np.expand_dims(img_data, 0)
            mean_vec = np.array([0.5, 0.5, 0.5])
            stddev_vec = np.array([0.5, 0.5, 0.5])
            norm_img_data = np.zeros(img_data.shape).astype('float32')
            for i in range(img_data.shape[1]):
                norm_img_data[:, i, :, :] = (img_data[:, i, :, :] / 255 -
                                             mean_vec[i]) / stddev_vec[i]
            if pre_result == None:
                pre_result = norm_img_data
            else:
                pre_result = np.concatenate((pre_result, norm_img_data),
                                            axis=0)
        return pre_result

    # transform the onnx reslt to detect result object
    def post_process(self, input_pil_arrays, boxes, scores):
        output_list = []
        boxes_data = []
        scores_data = []
        for image_id in range(len(input_pil_arrays)):

            image = input_pil_arrays[image_id]
            output_data = []
            for index in range(len(boxes[image_id])):
                if (scores[image_id][index][1]) > 0.8:
                    box = (boxes[image_id][index])
                    x1 = int(box[0] * image.size[0])
                    y1 = int(box[1] * image.size[1])
                    x2 = int(box[2] * image.size[0])
                    y2 = int(box[3] * image.size[1])
                    boxes_data.append([x1, y1, x2, y2])
                    scores_data.append(scores[image_id][index][1])

            nms_boxes, nms_scores = NMS(boxes_data, scores_data)
            output_list.append(nms_boxes)
        return output_list

    def label_frame(self, input_frames, input_pil_arrays, detect_result_list):
        from PIL import ImageDraw
        output_frame_list = []
        for index_frame in range(len(input_pil_arrays)):
            image = input_pil_arrays[index_frame]
            draw = ImageDraw.Draw(image)
            for index_box in range(len(detect_result_list[index_frame])):
                detect_result = detect_result_list[index_frame][index_box]
                draw.rectangle([
                    detect_result[0], detect_result[1], detect_result[2],
                    detect_result[3]
                ])
            del draw

            img = np.asarray(image)
            H420 = mp.PixelInfo(mp.kPF_YUV420P)
            rgb = mp.PixelInfo(mp.kPF_RGB24)

            frame = mp.Frame(mp.from_numpy(img), rgb)
            output_frame = VideoFrame(frame).reformat(H420)
            output_frame.pts = input_frames[index_frame].pts
            output_frame.time_base = input_frames[index_frame].time_base
            output_frame_list.append(output_frame)
        return output_frame_list

    def inference(self, input_tensor):
        input_tensor_ov = ov.Tensor(array=input_tensor, shared_memory=True)
        self.infer_request.set_input_tensor(input_tensor_ov)
        self.infer_request.start_async()
        self.infer_request.wait()
        score, boxes = self.infer_request.output_tensors
        return score.data, boxes.data

    def detect(self):
        frame_num = min(self.frame_cache_.qsize(), self.in_frame_num_)
        input_frames = []
        input_pil_arrays = []
        if frame_num == 0:
            return [], []
        for i in range(frame_num):
            vf = self.frame_cache_.get()
            input_frames.append(vf)

            rgb = mp.PixelInfo(mp.kPF_RGB24)
            numpy_vf = vf.reformat(rgb).frame().plane(0).numpy()
            input_pil_arrays.append(Image.fromarray(numpy_vf))

        input_tensor = self.pre_process(input_pil_arrays)
        scores, boxes = self.inference(input_tensor)
        detect_result_list = self.post_process(input_pil_arrays, boxes, scores)
        if self.label_frame_flag_ == 1:
            result_frames = self.label_frame(input_frames, input_pil_arrays,
                                             detect_result_list)
            return result_frames, detect_result_list

        return input_frames, detect_result_list

    def process(self, task):
        input_queue = task.get_inputs()[0]
        output_queue_size = len(task.get_outputs())

        while not input_queue.empty():
            pkt = input_queue.get()
            if pkt.timestamp == Timestamp.EOF:
                # we should done all frames processing in following loop
                self.eof_received_ = True
            if pkt.is_(VideoFrame):
                self.frame_cache_.put(pkt.get(VideoFrame))
        # detect processing
        while self.frame_cache_.qsize() >= self.in_frame_num_ or \
                self.eof_received_:
            data_list, extra_data_list = self.detect()
            for index in range(len(data_list)):
                # add sr output frame to task output queue
                pkt = Packet(data_list[index])
                pkt.timestamp = data_list[index].pts
                task.get_outputs()[0].put(pkt)
                # push output
                if (output_queue_size >= 2):
                    pkt = Packet(extra_data_list[index])
                    pkt.timestamp = data_list[index].pts
                    task.get_outputs()[1].put(pkt)

            # all frames processed, quit the loop
            if self.frame_cache_.empty():
                break

        # add eof packet to output
        if self.eof_received_:
            for key in task.get_outputs():
                task.get_outputs()[key].put(Packet.generate_eof_packet())
            Log.log_node(LogLevel.DEBUG, self.node_, 'output stream', 'done')
            task.set_timestamp(Timestamp.DONE)

        return ProcessResult.OK
