import onnxruntime as rt
import numpy as np
import bmf.hml.hmp as mp
import time
from bmf import Module, Log, LogLevel, InputType, ProcessResult, Packet, Timestamp, scale_av_pts, av_time_base, \
    BmfCallBackType, VideoFrame, AudioFrame
import sys

if sys.version_info.major == 2:
    from Queue import Queue
else:
    from queue import Queue


class onnx_sr(Module):
    def __init__(self, node=None, option=None):
        self.node_ = node
        self.option_ = option

        if option is None:
            Log.log(LogLevel.ERROR, "Option is none")
            return

        # model path
        if 'model_path' in option.keys():
            self.model_path_ = option['model_path']

        start_time = time.time()

        # load model
        # model path: "/Users/bytedance/code/v_lambda/models/sr/lab_videosr/v1.onnx"
        self.sess_ = rt.InferenceSession(self.model_path_)

        # Let's see the input name and shape.
        # in this example, we assume there is only one input
        self.input_name_ = self.sess_.get_inputs()[0].name
        self.input_shape_ = self.sess_.get_inputs()[0].shape
        self.input_type_ = self.sess_.get_inputs()[0].type
        Log.log(LogLevel.ERROR, "Input name", self.input_name_,
                "input shape:", self.input_shape_,
                "input type:", self.input_type_)

        # Let's see the output name and shape.
        self.output_name_ = self.sess_.get_outputs()[0].name
        self.output_shape_ = self.sess_.get_outputs()[0].shape
        self.output_type_ = self.sess_.get_outputs()[0].type
        Log.log(LogLevel.ERROR, "Output name", self.output_name_,
                "output shape:", self.output_shape_,
                "output type:", self.output_type_)

        # internal frame cache
        self.frame_cache_ = Queue()

        self.in_frame_num_ = 7
        self.out_frame_num_ = 3

        self.eof_received_ = False

        Log.log(LogLevel.ERROR, "Load model takes", (time.time() - start_time))

    def reset(self):
        # clear status
        self.eof_received_ = False
        while not self.frame_cache_.empty():
            self.frame_cache_.get()

    def sr_process(self):
        # convert video frame to nd array
        frame_num = min(self.frame_cache_.qsize(), self.in_frame_num_)
        input_frames = []
        input_nd_arrays = []
        for i in range(frame_num):
            vf = self.frame_cache_.queue[i]
            input_frames.append(vf)

            rgb = mp.PixelInfo(mp.kPF_RGB24)
            np_vf = vf.reformat(rgb).frame().plane(0).numpy()
            input_nd_arrays.append(np_vf)
        # for the last few frames, repeat the last frame
        for i in range(self.in_frame_num_ - frame_num):
            vf = input_frames[frame_num - 1]
            input_frames.append(vf)
            rgb = mp.PixelInfo(mp.kPF_RGB24)
            np_vf = vf.reformat(rgb).frame().plane(0).numpy()
            input_nd_arrays.append(np_vf)

        # combine 7 frames into one nd array as model input
        input_tensor = np.concatenate(input_nd_arrays, 2)
        input_tensor = np.expand_dims(input_tensor, 0)
        input_tensor = input_tensor.astype(np.uint8)
        Log.log_node(LogLevel.DEBUG, self.node_, "Start processing, input shape", input_tensor.shape)

        # predict
        output_tensor = self.sess_.run([self.output_name_], {self.input_name_: input_tensor})

        # split output tensor to 3 frames
        output_tensor = np.array(output_tensor).squeeze()
        output_tensor = np.array(np.split(output_tensor, 3, axis=2))
        Log.log_node(LogLevel.DEBUG, self.node_, "Finish processing, output shape", output_tensor.shape)

        # create output frames
        out_frames = []
        for i in range(self.out_frame_num_):
            H420 = mp.PixelInfo(mp.kPF_YUV420P)
            # convert nd array to video frame and convert rgb to yuv

            rgb = mp.PixelInfo(mp.kPF_RGB24)
            frame = mp.Frame(mp.from_numpy(output_tensor[i]), rgb)
            out_frame = VideoFrame(frame).reformat(H420)

            if self.frame_cache_.empty():
                break

            # dequeue input frame
            # copy frame attributes from (2 * i + 1)th input frame
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
                # we should done all frames processing in following loop
                self.eof_received_ = True
            elif pkt.is_(VideoFrame):
                self.frame_cache_.put(pkt.get(VideoFrame))

        # sr processing
        while self.frame_cache_.qsize() >= self.in_frame_num_ or \
                self.eof_received_:
            if self.frame_cache_.qsize() > 0:
                for frame in self.sr_process():
                    # add sr output frame to task output queue
                    pkt = Packet(frame)
                    pkt.timestamp = frame.pts
                    output_queue.put(pkt)
                    
            # all frames processed, quit the loop
            if self.frame_cache_.empty():
                break

        # add eof packet to output
        if self.eof_received_:
            output_queue.put(Packet.generate_eof_packet())
            Log.log_node(LogLevel.DEBUG, self.node_,
                         'output stream', 'done')
            task.timestamp = Timestamp.DONE

        return ProcessResult.OK
