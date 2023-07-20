#!/usr/bin/env python
# -*- coding: utf-8 -*

from bmf import Module, Log, Timestamp, ProcessResult, LogLevel, Packet, VideoFrame

from bmf.lib._bmf.sdk import ffmpeg
from bmf.hml import hmp as mp

import numpy as np


class Compositor:

    def __init__(self):
        self.slide_step = 5
        self.line_width = 10
        self.frame_width = 0
        self.frame_num = 0

    def merge(self, img_left, img_right, x_coord):
        img_out = np.copy(img_left)
        img_out[:, x_coord:, :] = img_right[:, x_coord:, :]

        # white line
        img_out[:, x_coord:x_coord + self.line_width, :] = 255

        return img_out

    def compose(self, img_left, img_right):
        if self.frame_width == 0:
            self.frame_width = img_left.shape[1]

        remainder = (self.frame_num * self.slide_step) % (2 * self.frame_width)
        x_coord = min(remainder, 2 * self.frame_width - remainder)

        out = self.merge(img_left, img_right, x_coord)

        self.frame_num += 1

        return out


class CompositionModule(Module):

    def __init__(self, node=None, option=None):
        self._node = node
        self.compositor = Compositor()

    def process(self, task):
        output_queue = task.get_outputs().get(0, None)

        current_pts = None
        current_timestamp = 0
        current_timebase = None

        # collect frames
        frame_list = []
        for (index, input_packets) in task.get_inputs().items():
            Log.log_node(
                LogLevel.DEBUG,
                self._node,
                "process input index: ",
                index,
            )

            num = 0
            while not input_packets.empty():
                pkt = input_packets.get()
                num += 1
                # process EOS
                if pkt.timestamp == Timestamp.EOF:
                    Log.log_node(LogLevel.INFO, task.get_node(),
                                 "Receive EOF.................")
                    if output_queue is not None:
                        output_queue.put(Packet.generate_eof_packet())
                    task.timestamp = Timestamp.DONE
                    return ProcessResult.OK

                frame = pkt.get(VideoFrame)

                # record
                if current_pts is None:
                    current_pts = frame.pts
                    current_timebase = frame.time_base
                    current_timestamp = pkt.timestamp

                rgb_frame = ffmpeg.reformat(frame,
                                            "rgb24").frame().plane(0).numpy()

                Log.log_node(
                    LogLevel.DEBUG,
                    self._node,
                    "index:",
                    index,
                    "get frame pts: ",
                    frame.pts,
                )
                frame_list.append(rgb_frame)

            Log.log_node(LogLevel.DEBUG, self._node, "index:", index,
                         "get frame num:", num)

        if len(frame_list) != 2:
            Log.log_node(
                LogLevel.WARNING,
                self._node,
                "frame list num incorrect, check framesync input",
            )
            return ProcessResult.OK

        out = self.compositor.compose(frame_list[0], frame_list[1])

        rgbformat = mp.PixelInfo(mp.kPF_RGB24)
        image = mp.Frame(mp.from_numpy(out), rgbformat)
        output_frame = VideoFrame(image)
        output_frame.pts = current_pts
        output_frame.time_base = current_timebase
        output_pkt = Packet(output_frame)
        output_pkt.timestamp = current_timestamp

        # output index 0 for test
        if output_queue is not None:
            output_queue.put(output_pkt)

        return ProcessResult.OK
