#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from bmf import Module, Log, LogLevel, InputType, ProcessResult, Packet, Timestamp, scale_av_pts, av_time_base, VideoFrame

from bmf.lib._bmf.sdk import Rational
import bmf.hml.hmp as mp


class copy_frame(Module):

    def __init__(self, node, option=None):
        self.node_ = node

    def process(self, task):
        output_packets = task.get_outputs()[0]
        input_packets = task.get_inputs()[0]

        while not input_packets.empty():
            pkt = input_packets.get()

            # process EOS
            if pkt.timestamp == Timestamp.EOF:
                Log.log_node(LogLevel.DEBUG, task.get_node(), "Receive EOF")
                output_packets.put(Packet.generate_eof_packet())
                task.timestamp = Timestamp.DONE
                return ProcessResult.OK

            frame = pkt.get(VideoFrame)
            width = frame.width
            height = frame.height
            video_frame = VideoFrame(width,
                                     height,
                                     mp.PixelInfo(mp.kPF_NV12),
                                     device="cuda")
            video_frame.copy_(frame)
            video_frame.time_base = frame.time_base
            video_frame.pts = frame.pts
            pkt = Packet(video_frame)
            pkt.timestamp = video_frame.pts
            output_packets.put(pkt)

        return ProcessResult.OK
