import sys
import torch
import torch.nn.functional as F
import PIL

sys.path.append("../../")
import bmf
from bmf import Log,LogLevel
import hmp as mp

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

def trt_detect_pre_process(infer_args):
    frame_cache = infer_args["frame_cache"]
    in_frame_num = infer_args["in_frame_num"]
    frame_num = min(frame_cache.qsize(), in_frame_num)
    input_frames = []
    pil_image_array = []
    torch_image_array = []
    resized_image_array = []
    
    width = infer_args["input_shapes"]["input"][3]
    height = infer_args["input_shapes"]["input"][2]
    if frame_num == 0:
        return [], []
    for i in range(frame_num):
        vf = frame_cache.queue[i]
        if (vf.frame().device() == mp.Device('cpu')):
            vf = vf.cuda()
        input_frames.append(vf)
        vf_image = vf.to_image(mp.kNHWC)
        torch_image = vf_image.image().data().torch()
        numpy_image = torch_image.cpu().numpy()
        torch_image_array.append(torch_image)
        pil_image_array.append(PIL.Image.fromarray(numpy_image))
        resized_image_array.append(F.interpolate(torch_image, size=(height, width), mode='bilinear'))
    for i in range(in_frame_num - frame_num):
        resized_image_array.append(resized_image_array[-1])
    
    input_tensor = torch.stack(resized_image_array).float()
    input_tensor = torch.permute(input_tensor, [0, 3, 1, 2])

    torch_mean = torch.empty(1, 3, 1, 1).fill_(0.5)
    torch_std = torch.empty(1, 3, 1, 1).fill_(0.5)

    input_tensor = (input_tensor - torch_mean) / torch_std

    input_dict = dict()
    input_dict["input"] = input_tensor.data_ptr()

    infer_args["input_dict"] = input_dict
    infer_args["image_array"] = pil_image_array
    infer_args["input_frames"] = input_frames

    return input_dict

def trt_detect_post_process(infer_args):
    frame_cache = infer_args["frame_cache"]
    out_frame_num = infer_args["out_frame_num"]
    output_dict = infer_args["output_dict"]
    image_array = infer_args["image_array"]
    input_frames = infer_args["input_frames"]
    output_frames = []
    detect_result_list = []

    boxes = output_dict["boxes"].cpu().numpy()
    scores = output_dict["scores"].cpu().numpy()

    for image_id in range(len(image_array)):
        image = image_array[image_id]
        boxes_num = boxes.shape[1]
        box_data = []
        for idx in range(boxes_num):
            if (scores[image_id, idx, 1] > 0.8):
                box = boxes[image_id, idx, :]
                x1 = int(box[0] * image.shape[0])
                y1 = int(box[1] * image.shape[1])
                x2 = int(box[2] * image.shape[0])
                y2 = int(box[3] * image.shape[1])
                detect_result = DetectResult(x1, y1, x2, y2, 1, scores[image_id, idx, 1])
                box_data.append(detect_result)
        detect_result_list.append(box_data)

        if infer_args["label_frame_flag"] == 1:
            draw = PIL.ImageDraw.Draw(image)
            for index_box in range(len(box_data)):
                detect_result = box_data[index_box]
                draw.rectangle([detect_result.x1_, detect_result.y1_, detect_result.x2_, detect_result.y2_])
            del draw
            numpy_image = np.asarray(image)
            H420 = mp.PixelInfo(mp.kRF_YUV420P)
            mp_image = mp.Image(mp.from_numpy(numpy_image.contiguous()), format=mp.KNHWC)
            out_frame = bmf.VideoFrame(mp_image).to_frame(H420)
            out_frame = out_frame.cpu()
            out_frame.ptr = input_frames[image_id].pts
            out_frame.time_base = input_frames[image_id].time_base
            output_frames.append(out_frame)
        else:
            output_frames.append(input_frames[image_id])

    return output_frames, detect_result_list

def main():
    graph = bmf.graph()

    
