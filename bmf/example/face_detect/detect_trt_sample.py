import sys
import torch
import numpy as np

sys.path.append("../../")
import bmf
from bmf import Log,LogLevel

def main():

    trt_face_detect = bmf.create_module("trt_face_detect", {
        "model_path": "version-RFB-640.engine",
        "label_to_frame": 1,
        "input_shapes": {"input": [1, 3, 480, 640]}
        })

    (
        bmf.graph()
            .decode({"input_path": "../files/face.mp4",
                     "video_params": {
                         "hwaccel": "cuda",
                         }})["video"]
            .module("trt_face_detect", pre_module=trt_face_detect)
            .encode(None, {"output_path": "./trt_out.mp4",
                           "video_params": {
                               "codec": "h264_nvenc",
                               "preset": "p6",
                               "tune": "hq",
                               "bit_rate": 5000000,
                               }})
            .run()
    )

if __name__ == "__main__":
    main()
