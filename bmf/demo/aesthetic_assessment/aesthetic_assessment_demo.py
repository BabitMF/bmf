#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import bmf
import cv2, os, sys


def get_duration(video_path):
    capture = cv2.VideoCapture(video_path)
    fps = capture.get(
        cv2.CAP_PROP_FPS)  # OpenCV2 version 2 used "CV_CAP_PROP_FPS"
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps
    capture.release()
    return duration


def segment_decode_ticks(video_path,
                         seg_dur=4.0,
                         lv1_dur_thres=24.0,
                         max_dur=1000):
    """
    bmf module new decode duration ticks
    - 0 < Duration <= 24s, 抽帧间隔r=1, 抽帧0~24帧
    - 24s < Duration <= 600s 分片抽取, 抽帧间隔r=1, 抽帧24帧
        - 6个4s切片, 共计6x4=24帧
    - duration > 600s, 分8片抽帧r=1, 抽帧数量32帧
        - (600, inf), 8个4s切片, 共计8x4=32帧
    最大解码长度 max_dur: 1000s
    """
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
        seg_intev = (min(duration, max_dur) - seg_num * seg_dur) / (seg_num -
                                                                    1)
        for s_i in range(seg_num):
            seg_init = s_i * (seg_dur + seg_intev)
            seg_end = seg_init + seg_dur
            duration_ticks.extend([round(seg_init, 3), round(seg_end, 3)])
    return {"durations": duration_ticks}


if __name__ == "__main__":
    input_path = "bbb_360_20s.mp4"
    outp_path = "res2.json"

    option = dict()
    option["output_path"] = outp_path

    # check model path
    model_path = "../../models/aes_transonnx_update3.onnx"
    if not os.path.exists(model_path):
        print(
            "please download model first, use 'wget https://github.com/BabitMF/bmf/releases/download/files/models.tar.gz && tar zxvf models.tar.gz' "
        )
        exit(0)

    option["model_path"] = model_path
    print("option", option)
    duration_segs = segment_decode_ticks(input_path)
    decode_params = {
        "input_path": input_path,
        "video_params": {
            "extract_frames": {
                "fps": 1
            }
        },
    }
    decode_params.update(duration_segs)
    print("decode_params", decode_params)
    # module process

    py_module_path = os.path.abspath(
        os.path.dirname(os.path.dirname("__file__")))
    py_entry = "aesmod_module.BMFAesmod"
    print(py_module_path, py_entry)

    streams = bmf.graph().decode(decode_params)
    video_stream = streams["video"].module("aesmod_module", option,
                                           py_module_path, py_entry)
    video_stream.run()
