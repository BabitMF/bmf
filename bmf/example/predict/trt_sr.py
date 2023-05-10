import numpy as np
import torch
import sys

sys.path.append("../../")
sys.path.append("../tensorrt/")
import bmf
import hmp as mp

def trt_sr_pre_process(infer_args):
    frame_cache = infer_args["frame_cache"]
    in_frame_num = infer_args["in_frame_num"]
    frame_num = min(frame_cache.qsize(), in_frame_num)
    input_frames = []
    input_torch_array = []
    for i in range(frame_num):
        vf = frame_cache.queue[i]
        if (vf.frame().device() == mp.Device('cpu')):
            vf = vf.cuda()
        input_frames.append(vf)
        vf_image = vf.to_image(mp.kNHWC)
        input_torch_array.append(vf_image.image().data().torch())
    for i in range(in_frame_num - frame_num):
        input_torch_array.append(input_torch_array[-1])

    input_tensor = torch.concat(input_torch_array, 2)

    input_dict = dict()
    input_dict["input:0"] = input_tensor.data_ptr()
    infer_args["input_dict"] = input_dict

def trt_sr_post_process(infer_args):
    frame_cache = infer_args["frame_cache"]
    out_frame_num = infer_args["out_frame_num"]

    output_dict = infer_args["output_dict"]
    output_tensor = output_dict["output:0"]

    output_tensor_torch = output_tensor.torch()
    output_tensor_torch = torch.squeeze(output_tensor_torch)
    output_tensor_torch = torch.split(output_tensor_torch, out_frame_num, dim=2)

    out_frames = []

    for i in range(out_frame_num):
        H420 = mp.PixelInfo(mp.kPF_YUV420P)
        image = mp.Image(mp.from_torch(output_tensor_torch[i].contiguous()), format=mp.kNHWC)
        out_frame = bmf.VideoFrame(image).to_frame(H420)
        out_frame = out_frame.cpu()

        if frame_cache.empty():
            break

        input_frame = frame_cache.get()
        out_frame.pts = input_frame.pts
        out_frame.time_base = input_frame.time_base

        out_frames.append(out_frame)

    return out_frames

def trt_rs():

    input_video_path = "../files/img_s.mp4"
    output_video_path = "out.mp4"

    (
        bmf.graph()
            .decode({"input_path": input_video_path,
                    "video_params": {
                        "hwaccel": "cuda",
                    }})['video']
            .module("trt_inference", {"model_path": "v1_orig.engine",
                                    "input_shapes": {"input:0": [1, 360, 640, 21]},
                                    "pre_process": "trt_sr.trt_sr_pre_process",
                                    "post_process": "trt_sr.trt_sr_post_process",
                                    "in_frame_num": 7,
                                    "out_frame_num": 3})
            .encode(None, {"output_path": output_video_path,
                        "video_params": {
                            "codec": "h264_nvenc",
                            #"pix_fmt": "cuda",
                            "preset": "p6",
                            "tune": "hq",
                            "bit_rate": 5000000,
                        }})
            .run()
    )

if __name__ == '__main__':
    trt_rs()


