#!/usr/bin/env python
# -*- coding: utf-8 -*-

from bmf import (
    Module,
    Log,
    LogLevel,
    ProcessResult,
    Packet,
    Timestamp,
    VideoFrame,
)
import json
import time


class switch_module(Module):
    main_stream = 0
    switch_stream = 1
    def __init__(self, node, option=None):
        self.input_packets = []
        # None means wait
        # 1 means passthrough
        # 0 means not passthrough
        self.switch = None
        self.process_done = False
        self.main_eof_skip = False
        self.version = self.get_version()

    def process_pkt(self, task, pkt):
        if self.switch is None:
            self.input_packets.append(pkt)
            return
        output_queue = task.get_outputs().get(0, None)
        if self.switch == 0:
            self.process_done = True
            if output_queue is not None:
                output_queue.put(Packet.generate_eof_packet())
            task.timestamp = Timestamp.DONE

        if self.switch == 1:
            for packet in self.input_packets:
                if output_queue is not None:
                    output_queue.put(packet)
            self.input_packets = []
            if output_queue is not None:
                output_queue.put(pkt)

    def process_main_eof(self, task):
        Log.log(LogLevel.INFO, "process main eof, switch: ", self.switch)
        if self.switch is None:
            return True
        output_queue = task.get_outputs().get(0, None)

        if self.switch == 1:
            for packet in self.input_packets:
                if output_queue is not None:
                    output_queue.put(packet)
            self.input_packets = []

        self.process_done = True
        if output_queue is not None:
            output_queue.put(Packet.generate_eof_packet())
        task.timestamp = Timestamp.DONE
        return False


    def process(self, task):
        output_queue = task.get_outputs().get(0, None)
        main_queue = task.get_inputs().get(0, None)
        switch_queue = task.get_inputs().get(1, None)

        if self.process_done:
            return ProcessResult.OK

        #save main_queue
        while not main_queue.empty():
            pkt = main_queue.get()
            if pkt.timestamp == Timestamp.EOF:
                self.main_eof_skip = self.process_main_eof(task)
            else:
                self.process_pkt(task,pkt)

        while not switch_queue.empty() and self.switch is None:
            pkt = switch_queue.get()
            if pkt.timestamp == Timestamp.EOF:
                Log.log(LogLevel.INFO, "get eof, set switch 0")
                self.switch = 0
            else:
                Log.log(LogLevel.INFO, "get pkt, set switch 1")
                self.switch = 1

        if self.main_eof_skip and self.switch is not None:
            Log.log(LogLevel.INFO, "process main eof last time")
            self.process_main_eof(task)

        return ProcessResult.OK

    def get_version(self):
        return "v1"