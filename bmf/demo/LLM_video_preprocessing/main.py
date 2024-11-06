#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from media_info import MediaInfo
from split import get_split_info
import torch
from clip_process import ClipProcess
import argparse
import os

logger = logging.getLogger('main')

scene_thres = 0.3

def get_timeline_list(pts_list, last):
    current = 0
    timelines = []
    for pts in pts_list:
        pts = pts/1000000
        if pts > current:
            timelines.append([current,pts])
        current = pts
    # last
    if last > current:
        timelines.append([current,last])
    return timelines

def video_processing_demo(input_file, mode, config):
    media_info = MediaInfo("ffprobe", input_file)
    duration = media_info.video_duration()
    logger.info(f"duration:{duration}")

    pts_list = get_split_info(input_file, scene_thres)
    timelines = get_timeline_list(pts_list, duration)
    logger.info(f"timelines:{timelines}")
    cp = ClipProcess(input_file, timelines, mode, config)
    cp.process()

def demo_run(args):
    if args.input_file == '':
        print("input file needed")
        return -1
    model_path = "../../models/aes_transonnx_update3.onnx"
    if not os.path.exists(model_path):
        print(
            "please download model first, use 'wget https://github.com/BabitMF/bmf/releases/download/files/models.tar.gz && tar zxvf models.tar.gz -C ../../' "
        )
        exit(0)
    torch.set_num_threads(4)
    mode = "ocr_crop,aesmod_module"
    config = {"output_configs":[{"res":"orig", "type":"jpg"}, {"res":"480", "type":"mp4"}]}
    video_processing_demo(args.input_file, mode, config)

if __name__ == '__main__':
    logger.setLevel(logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, help="input local file", default = '')
    args = parser.parse_args()
    demo_run(args)
