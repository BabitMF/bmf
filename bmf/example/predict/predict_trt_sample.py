import sys

sys.path.append("../../")

import bmf
from bmf import LogLevel, Log

if __name__ == "__main__":
    Log.set_log_level(LogLevel.ERROR)

    input_video_path = "../files/img_s.mp4"
    output_video_path = "out.mp4"

    # create trt sr module once
    trt_sr_mod = bmf.create_module("trt_sr", {"model_path": "v1_orig.engine",
                                              "input_shapes": {"input:0": [1, 360, 640, 21]},
                                              "in_frame_num": 7,
                                              "out_frame_num": 3})
    (
        bmf.graph()
            .decode({"input_path": input_video_path,
                     "video_params": {
                         "hwaccel": "cuda",
                         }})["video"]
            .module("trt_sr", pre_module=trt_sr_mod)
            .encode(None, {"output_path": output_video_path,
                           "video_params": {
                               "codec": "h264_nvenc",
                               "preset": "p6",
                               "tune": "hq",
                               "bit_rate": 5000000,
                               }})
            .run()
    )
