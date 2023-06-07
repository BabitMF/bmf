#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
from fractions import Fraction
from bmf import Module, ProcessResult, Packet, av_time_base, Log, LogLevel

from pts_generator import PtsGenerator


class WallClock(Module):
    def __init__(self, node, option=None):
        option = option if option else dict()
        self.video_step = Fraction(option.get("video_step", 1))
        self.audio_step = Fraction(option.get("audio_step", 1))
        self.node_ = node

        self.pts_generator_list = []

    def process(self, task):
        now = time.time()
        if len(self.pts_generator_list) < 2:
            self.pts_generator_list.append(PtsGenerator(now, self.video_step))
            self.pts_generator_list.append(PtsGenerator(now, self.audio_step))

        for index, gen in enumerate(self.pts_generator_list):
            output_queue = task.get_outputs().get(index, None)
            for pts_sec in gen.generate(now):
                pts = int(float(pts_sec / av_time_base))
                bmf_pkt = Packet(0)
                bmf_pkt.timestamp = pts
                if output_queue:
                    output_queue.put(bmf_pkt)

        time.sleep(0.01)

        return ProcessResult.OK
