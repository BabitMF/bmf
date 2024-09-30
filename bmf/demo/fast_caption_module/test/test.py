import bmf
import numpy as np
import torch
import argparse
import logging
from bmf import GraphMode, BMFAVPacket, Packet
import pyarrow as pa
import pyarrow.parquet as pq
import subprocess
import time
from PIL import Image
from bmf.lib._bmf.sdk import ffmpeg
IMG_COLUMN_NAME = "image_bytes"
num_frames = 4
params = {
    "model_path": "../../../3rd_party/llava",
    "num_frames": num_frames,
    "batch_size": 4,
    "cap_prompt": "Expand what's the video shows",
    "gpu": 2
}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--gpus", type=list, default=0)
    parser.add_argument("-l", "--lib", default="libbcm",
                        help="lib name, libbcm or libbcm_d")
    parser.add_argument("-v", "--videos", default="example/videos",
                        help="videos path for caption")

    args = parser.parse_args()
    return args

import glob

if __name__ == '__main__':
    args = parse_args()

    module = bmf.bmf_sync.sync_module(
        {"name": "caption", "type": "c++", "path": f"build/output/lib/{args.lib}.so", 
         "entry": f"{args.lib}:caption"}, params, [0], [0]
    )
    module.init()

    test_case = glob.glob(args.videos + "/*")
    condition = True
    decoders = []
    for index, video_path in enumerate(test_case):
        decoder = bmf.bmf_sync.sync_module("c_ffmpeg_decoder",
                                            {
                                                "input_path": video_path,
                                            }, [0], [0])
        decoders.append(decoder)

    start = time.time()
    task_lists = []
    for decoder in decoders:
        task_list = []
        condition = True
        while condition:
            pkt = bmf.bmf_sync.process(decoder, {})[0][0]
            try:
                task_list.append(pkt[0].get(bmf.VideoFrame))    
            except:
                break
        print(len(task_list))
        task_list = task_list[::round(len(task_list) / num_frames)]
        print(len(task_list))
        task_lists.append(task_list)
    
    for task_list in task_lists:
        for i, videoframe in enumerate(task_list):
            videoframe = ffmpeg.reformat(videoframe, "rgb24")
            pkt = bmf.Packet(videoframe)
            _ = bmf.bmf_sync.process(module, {0: [pkt]})[0]

    print(f"duration is {time.time() - start} s")
    
