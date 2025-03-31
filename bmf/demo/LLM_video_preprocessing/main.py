#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from media_info import MediaInfo
from split import get_split_info
import torch
from clip_process import ClipProcess
import argparse
import os
import json

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

def video_processing_demo(input_file, modes, config):
    media_info = MediaInfo("ffprobe", input_file)
    duration = media_info.video_duration()
    logger.info(f"duration:{duration}")

    pts_list = get_split_info(input_file, scene_thres)
    timelines = get_timeline_list(pts_list, duration)
    logger.info(f"timelines:{timelines}")
    cp = ClipProcess(input_file, timelines, modes, config)
    cp.process()

def load_config(path):
    if os.path.exists(path):
        with open(path, 'r') as file:
            return json.load(file)
    else:
        return {}

def demo_run(args):
    torch.set_num_threads(4)
    config = load_config(args.config) if args.config else {}
    if not os.path.exists(args.input_file):
        print(f"Input file does not exist {args.input_file}")
        return -1
    if "mode" not in config:
        print(f"Mode not specified in {args.config}")
        return -1

    modes = set(config["mode"].split(","))
    if "aesmod_module" in modes and not os.path.exists("../../models/aes_transonnx_update3.onnx"):
        print("To use aesmod_module, download the model first", \
              "'wget https://github.com/BabitMF/bmf/releases/download/files/models.tar.gz && tar zxvf models.tar.gz -C ../../' ")
        return -1

    video_processing_demo(args.input_file, modes, config)

if __name__ == '__main__':
    logger.setLevel(logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help="path to JSON config file")
    parser.add_argument('input_file', type=str, help="path to video file")
    args = parser.parse_args()
    demo_run(args)
