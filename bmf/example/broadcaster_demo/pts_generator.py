#!/usr/bin/env python
# -*- coding: utf-8 -*-


class PtsGenerator:
    def __init__(self, start_time, step):
        self.start_time = start_time
        self.step = step
        self.num = 0

    def generate(self, now):
        pts_list = []
        diff = now - self.start_time
        while True:
            pts = self.num * self.step
            if pts <= diff:
                pts_list.append(pts)
                self.num += 1
            else:
                break
        return pts_list
