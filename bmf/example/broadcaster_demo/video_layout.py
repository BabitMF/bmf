#!/usr/bin/env python
# encoding: utf-8
"""
Created on 2020-11-25

@author: hejianqiang

modified by huheng on 2023-02-08

"""
import os
import sys
import time
import cv2
import json

# print(os.sys.path)
# sys.path.append("/opt/tiger/bmf")
# sys.path.append("/opt/tiger/bmf/bmf/lib")
# sys.path.append("/opt/tiger/bmf/bmf/python_builtins")
# sys.path.append("/opt/tiger/bmf/bmf/python_sdk")
# print(os.sys.path)
import bmf.hml.hmp as mp

from cv_frame import bg_overlay_frames
from bmf import (
    Module,
    Log,
    LogLevel,
    InputType,
    ProcessResult,
    Packet,
    Timestamp,
    VideoFrame,
    av_time_base,
)
from bmf.lib._bmf.sdk import Rational
from bmf.lib._bmf.sdk import ffmpeg


class video_layout(Module):
    def __init__(self, node=None, option=None):
        self._node = node
        self.option = option if option else dict()
        self._eof_received = False
        self.g_frame_num = 0
        self.g_monitor_change = {
            "frame_num": None,
            "layout_mode": None,
            "layout_location": None,
            "main_stream_idx": None,
            "layout_extra": None,
        }
        # self.nd_static_frame = VideoFrame(1280, 720, "rgb24").to_ndarray(format="rgb24")

        # if self.nd_static_frame is None:
        #    print("not a valide frame")

        self.option["rotation"] = []
        self._rotation_idx_table = dict()

    def dynamic_reset(self, opt_reset=None):
        Log.log_node(
            LogLevel.INFO,
            self._node,
            "opt_reset type:",
            type(opt_reset),
            "opt_reset: ",
            opt_reset,
        )
        if opt_reset is None:
            return
        if self.option is None:
            self.option = dict()
        for (para, value) in opt_reset.items():
            self.option[para] = value
        Log.log_node(
            LogLevel.INFO,
            self._node,
            "opt_reset:",
            opt_reset,
            "self.option: ",
            self.option,
        )
        """
        'layout_mode': '',  # speaker  gallery
        'crop_mode': '',  # pad crop scale
        'layout_location': '',
        'interspace': 0,
        'main_stream_idx': 0,
        "width": 0,
        "height": 0,
        "background_color": "#000000",
        "position": [0,2,1,3]
        "ratation": [0,90,-90,0]
        """

    def do_overlay(self, frame_list):
        overlay_list = []
        for (index, pkt) in frame_list:
            if index == -1:
                continue

            Log.log_node(LogLevel.DEBUG, self._node, "video frame to knhwc.......")
            video_frame = pkt.get(VideoFrame)
            #frame = video_frame.to_image(mp.kNHWC).image().numpy()

            #use ffmpeg
            frame = ffmpeg.reformat(video_frame, "rgb24").frame().plane(0).numpy()

            Log.log_node(LogLevel.DEBUG, self._node, "video frame to rgb24 done")
            # frame = video_frame.to_ndarray(format="rgb24")
            overlay_list.append(frame)

        raw_frame = bg_overlay_frames(overlay_list, self.option, self.g_monitor_change)
        return raw_frame

    def process(self, task):
        output_queue = task.get_outputs().get(0, None)

        for (input_idx, pkt_queue) in task.get_inputs().items():
            while not pkt_queue.empty():
                # bmf_pkt is frame list [(stream_id, pkt)]
                timestamp = 0
                bmf_pkt = pkt_queue.get()
                # frame_list = bmf_pkt.get_data()
                frame_list = bmf_pkt.get(list)
                Log.log_node(
                    LogLevel.DEBUG,
                    self._node,
                    "get frame list len: ",
                    len(frame_list),
                )

                if frame_list:
                    timestamp = frame_list[0][1]

                Log.log_node(LogLevel.DEBUG, self._node, "do video overlay.......")
                raw_frame = self.do_overlay(frame_list)
                Log.log_node(LogLevel.DEBUG, self._node, "do video overlay done")

                rgbformat = mp.PixelInfo(mp.kPF_RGB24)
                image = mp.Frame(mp.from_numpy(raw_frame), rgbformat)

                #H420 = mp.PixelInfo(mp.kPF_YUV420P)
                #video_frame = VideoFrame(image).to_frame(H420)
                #vf = VideoFrame(image)

                #video_frame = ffmpeg.reformat(vf, "yuv420p")
                video_frame = VideoFrame(image)
                Log.log_node(LogLevel.DEBUG, self._node, "do video format change done")
                # video_frame = VideoFrame.from_ndarray(
                #    raw_frame, format="rgb24"
                # ).reformat(format="yuv420p")


                video_frame.pts = timestamp
                video_frame.time_base = Rational(1, 1000000)
                output_pkt = Packet(video_frame)
                output_pkt.timestamp = timestamp
                if output_queue is not None:
                    output_queue.put(output_pkt)
                Log.log_node(
                    LogLevel.DEBUG,
                    self._node,
                    "output video frame, pts: ",
                    timestamp,
                )

        return ProcessResult.OK


if __name__ == "__main__":
    module = video_layout()
