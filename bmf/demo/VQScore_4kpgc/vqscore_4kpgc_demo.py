#!/usr/bin/env python
# -*- coding: utf-8 -*-
import bmf
import os
import sys
import cv2

def get_width_and_height(video_path):
    capture = cv2.VideoCapture(video_path)
    height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
    capture.release()
    return int(width), int(height)

def get_duration(video_path):
    capture = cv2.VideoCapture(video_path)
    fps = capture.get(cv2.CAP_PROP_FPS)      # OpenCV2 version 2 used "CV_CAP_PROP_FPS"
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps
    capture.release()
    return duration

def segment_decode_ticks(video_path, seg_dur=4.0, lv1_dur_thres=24.0, max_dur=1000):
    '''
        bmf module new decode duration ticks
        - 0 < Duration <= 24s, 抽帧间隔r=1, 抽帧0~24帧
        - 24s < Duration <= 600s 分片抽取, 抽帧间隔r=1, 抽帧24帧
            - 6个4s切片, 共计6x4=24帧
        - duration > 600s, 分8片抽帧r=1, 抽帧数量32帧
            - (600, inf), 8个4s切片, 共计8x4=32帧
        最大解码长度 max_dur: 1000s
    '''
    duration = get_duration(video_path)
    duration_ticks = []
    if duration < lv1_dur_thres:
        return dict()
    elif duration <= 600:  # medium duration
        seg_num = 6
        seg_intev = (duration - seg_num * seg_dur) / (seg_num - 1)
        if seg_intev < 0.5:
            duration_ticks.extend([0, duration])
        else:
            for s_i in range(seg_num):
                seg_init = s_i * (seg_dur + seg_intev)
                seg_end = seg_init + seg_dur
                duration_ticks.extend([round(seg_init, 3), round(seg_end, 3)])
    else:  # long duration
        seg_num = 8
        seg_intev = (min(duration, max_dur) - seg_num * seg_dur) / (seg_num - 1)
        for s_i in range(seg_num):
            seg_init = s_i * (seg_dur + seg_intev)
            seg_end = seg_init + seg_dur
            duration_ticks.extend([round(seg_init, 3), round(seg_end, 3)])
    return {'durations': duration_ticks}


if __name__=='__main__':
    #input_path='source/20f72ebc978c4b06830e23adee6b6ff7'
    input_path='source/VD_0290_00405.png'
    out_path='result/20f72ebc978c4b06830e23adee6b6ff7.json'

    # check model path
    model_path = "models/vqa_4kpgc_1.onnx"
    if not os.path.exists(model_path):
        print(
            "please download model first"
        )
        exit(0)


    option = dict()
    option['output_path'] = out_path
    option['width'], option['height'] = get_width_and_height(input_path)

    duration_segs = segment_decode_ticks(input_path)
    decode_params = {'input_path': input_path,
                        'video_params': {'extract_frames': {'fps': 1}}}
    decode_params.update(duration_segs)

    # module process
    streams = bmf.graph().decode(decode_params)
    py_module_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    py_entry = 'vqscore_4kpgc_module.BMFVQA_4kpgc'
    video_stream = streams['video'].module('vqscore_4kpgc_module', option,
                                            py_module_path, py_entry)
    video_stream.upload().run()












    
