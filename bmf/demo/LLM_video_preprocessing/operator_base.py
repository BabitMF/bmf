#!/usr/bin/env python
# -*- coding: utf-8 -*-

from bmf import Module, Timestamp, ProcessResult


class BaseModule(Module):
    def __init__(self, node, option):
        self.node_ = node
        self.result_path = (
            option.get("result_path")
            if option is not None and "result_path" in option
            else f"{node}.json"
        )

    def process(self, task):
        for _, input_packets in task.get_inputs().items():
            while not input_packets.empty():
                pkt = input_packets.get()
                if pkt.timestamp == Timestamp.EOF:
                    self.on_eof(task, pkt)
                else:
                    self.on_pkt(task, pkt)
        return ProcessResult.OK

    def on_eof(self, task, pkt):
        return

    def on_pkt(self, task, pkt):
        return
