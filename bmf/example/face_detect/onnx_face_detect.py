import onnxruntime as rt
import numpy as np
import time
import sys
if sys.version_info.major == 2:
    from Queue import Queue
else:
    from queue import Queue

from PIL import Image
from bmf import Module, Log, LogLevel, InputType, ProcessResult, Packet, Timestamp, scale_av_pts, av_time_base, \
    BmfCallBackType, VideoFrame, AudioFrame
import threading


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

class onnx_face_detect(Module):
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
        start_time = time.time()

        # load model
        self.sess_ = rt.InferenceSession(self.model_path_)

        self.output_names_=[]
        for output in self.sess_.get_outputs():
            self.output_names_.append(output.name)

        self.input_names_ = []
        self.input_shapes_ =[]
        for input in self.sess_.get_inputs():
            self.input_names_.append(input.name)
            self.input_shapes_.append(input.shape)

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

    def pre_process(self,image_list):

        image_size=(self.input_shapes_[0][3],self.input_shapes_[0][2])
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
                norm_img_data[:, i, :, :] = (img_data[:, i, :, :] / 255 - mean_vec[i]) / stddev_vec[i]
            if pre_result == None:
                pre_result = norm_img_data
            else:
                pre_result=np.concatenate((pre_result,norm_img_data),axis=0)
        return pre_result

    # transform the onnx reslt to detect result object
    def post_process(self,input_pil_arrays,boxes,scores):
        output_list=[]
        for image_id in range(len(input_pil_arrays)):

            image = input_pil_arrays[image_id]
            output_data = []
            for index in range(len(boxes[0])):
                if (scores[image_id][index][1]) > 0.8:
                    box = (boxes[image_id][index])
                    x1=int(box[0] * image.size[0])
                    y1=int(box[1] * image.size[1])
                    x2=int(box[2] * image.size[0])
                    y2=int(box[3] * image.size[1])
                    detect_result = DetectResult(x1, y1, x2, y2, 1, scores[image_id][index][1])
                    output_data.append(detect_result)
            output_list.append(output_data)
        return output_list

    # overlay info to frame
    def label_frame(self,input_frames,input_pil_arrays,detect_result_list):
        from PIL import ImageDraw
        output_frame_list=[]
        for index_frame in range(len(input_pil_arrays)):
            image = input_pil_arrays[index_frame]
            draw = ImageDraw.Draw(image)
            for index_box in range(len(detect_result_list[index_frame])):
                detect_result = detect_result_list[index_frame][index_box]
                draw.rectangle([detect_result.x1_, detect_result.y1_, detect_result.x2_,detect_result.y2_])
            del draw
            img = np.asarray(image)
            output_frame = VideoFrame.from_ndarray(img,"rgb24")
            output_frame.pts = input_frames[index_frame].pts
            output_frame.time_base = input_frames[index_frame].time_base
            output_frame_list.append(output_frame)
        return output_frame_list

    def detect(self):
        frame_num = min(self.frame_cache_.qsize(), self.in_frame_num_)
        input_frames = []
        input_pil_arrays = []
        if frame_num==0:
            return [],[]
        for i in range(frame_num):
            frame = self.frame_cache_.get()
            input_frames.append(frame)
            input_pil_arrays.append(Image.fromarray(np.uint8(frame.to_ndarray(format="rgb24"))))

        input_tensor = self.pre_process(input_pil_arrays)
        scores,boxes = self.sess_.run(self.output_names_, {self.input_names_[0]: input_tensor})
        detect_result_list= self.post_process(input_pil_arrays,boxes,scores)
        if self.label_frame_flag_==1:
            result_frames=self.label_frame(input_frames,input_pil_arrays,detect_result_list)
            return result_frames,detect_result_list


        return input_frames,detect_result_list

    def process(self, task):
        input_queue = task.get_inputs()[0]
        output_queue_size = len(task.get_outputs())

        while not input_queue.empty():
            pkt = input_queue.get()
            if pkt.get_timestamp() == Timestamp.EOF:
                # we should done all frames processing in following loop
                self.eof_received_ = True
            if pkt.get_data() is not None:
                self.frame_cache_.put(pkt.get_data())
        # detect processing
        while self.frame_cache_.qsize() >= self.in_frame_num_ or \
                self.eof_received_:
            data_list,extra_data_list = self.detect()
            for index in range(len(data_list)):
                # add sr output frame to task output queue
                pkt = Packet()
                pkt.set_timestamp(data_list[index].pts)
                pkt.set_data(data_list[index])
                task.get_outputs()[0].put(pkt)
                # push output
                if(output_queue_size>=2):
                    pkt = Packet()
                    pkt.set_timestamp(data_list[index].pts)
                    pkt.set_data(extra_data_list[index])
                    task.get_outputs()[1].put(pkt)

            # all frames processed, quit the loop
            if self.frame_cache_.empty():
                break

        # add eof packet to output
        if self.eof_received_:
            for key in task.get_outputs():
                task.get_outputs()[key].put(Packet.generate_eof_packet())
            Log.log_node(LogLevel.DEBUG, self.node_,
                        'output stream', 'done')
            task.set_timestamp(Timestamp.DONE)

        return ProcessResult.OK
