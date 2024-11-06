#!/usr/bin/env python
# -*- coding: utf-8 -*-

from dataclasses import dataclass, field
from operator_base import BaseModule
import easyocr
from bmf import VideoFrame, av_time_base, Log, LogLevel, Packet, Timestamp
from bmf.lib._bmf.sdk import ffmpeg
import json
import numpy as np

def point_in_area(point, area_coord):
    px,py=point
    x0, y0, x1, y1 = area_coord
    if px > x0 and px < x1 and py > y0 and py < y1:
        return True
    return False

def hole_in_area(hole, area_coord):
    for p in hole:
        if point_in_area(p, area_coord):
            return True
    return False

def split_area_by_hole(hole, area_coord):
    res = []
    if not hole_in_area(hole, area_coord):
        res.append(area_coord)
        return res

    hole_x0 = hole[0][0]
    hole_x1 = hole[1][0]
    hole_y0 = hole[0][1]
    hole_y1 = hole[2][1]

    x0, y0, x1, y1 = area_coord
    if hole_x0 > x0:
        res.append((x0, y0, hole_x0, y1))
    if hole_x1 < x1:
        res.append((hole_x1, y0, x1, y1))
    if hole_y0 > y0:
        res.append((x0, y0, x1, hole_y0))
    if hole_y1 < y1:
        res.append((x0, hole_y1, x1, y1))
    return res

def get_area(area_coord):
    x0, y0, x1, y1 = area_coord
    return (x1-x0)*(y1-y0)

def calc_max_area_(blackholes, area_coord, max_area_coord, max_area_thres):
    holes = blackholes[0:]
    max_area = get_area(max_area_coord)
    if len(holes) == 0:
        area = get_area(area_coord)
        if max_area < area:
            max_area = area
            max_area_coord = area_coord
    else:
        hole = holes[0]
        sub_holes = holes[1:]
        res = split_area_by_hole(hole, area_coord)
        for sub_area_coord in res:
            if get_area(sub_area_coord) <= max_area:
                continue
            max_sub_area_coord = calc_max_area_(sub_holes, sub_area_coord, max_area_coord, max_area_thres)
            area = get_area(max_sub_area_coord)
            if max_area < area:
                max_area = area
                max_area_coord = max_sub_area_coord
    return max_area_coord

# return the maximum area without blackholes
# stop calculate if area ratio is ls less than ratio_thres
def calc_max_area(blackholes, width, height, ratio_thres):
    max_area_coord = (0,0,0,0)
    max_area_thres = int(width*height*ratio_thres)
    max_area_coord = calc_max_area_(blackholes, (0, 0, width, height), max_area_coord, max_area_thres)
    x0, y0, x1, y1 = max_area_coord
    res = (x0/width, y0/height, x1/width, y1/height)
    if get_area(max_area_coord) > max_area_thres:
        return True, res
    return False, res

@dataclass
class ocr_crop_params:
    nms_remain_area_ratio_thres: float = field(default=0)
    word_prob_threshold: float = field(default=0.3)
    sample_every_seconds: float = field(default=1)


class ocr_crop(BaseModule):
    def __init__(self, node, option):
        super().__init__(node, option)
        self.reader = easyocr.Reader(["ch_sim", "en"])
        self.params = ocr_crop_params(
            **option["params"] if option is not None and "params" in option else {}
        )
        self.reset()

    def reset(self):
        self.input_packets = []
        self.ocr_results = []
        self.frame_width = None
        self.frame_height = None
        self.last_processed_frame_timestamp = None

    def write_result(self, ok, nms_crop_box):
        output = {}
        output["video_width"] = self.frame_width
        output["video_height"] = self.frame_height
        output["nms_remain_area_ratio_thres"] = self.params.nms_remain_area_ratio_thres
        output["word_prob_threshold"] = self.params.word_prob_threshold
        output["get_max_area"] = ok
        output["nms_crop_box"] = nms_crop_box
        res = {}
        res["result"] = output

        with open(self.result_path, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=4)

    def process_ocr_results(self):
        """
        ocr_results = [(
            [
                [np.int32(86), np.int32(80)],
                [np.int32(134), np.int32(80)],
                [np.int32(134), np.int32(128)],
                [np.int32(86), np.int32(128)],
            ],
            "西",
            np.float64(0.6629598563364745),
        )]
        """

        ocr_areas = []
        for frame_results in self.ocr_results:
            for res in frame_results:
                # nms points coordinate
                if res[2] > self.params.word_prob_threshold:
                    ocr_areas.append(res[0])
        # debug
        print(ocr_areas)
        return calc_max_area(
            ocr_areas,
            self.frame_width,
            self.frame_height,
            self.params.nms_remain_area_ratio_thres,
        )

    def on_eof(self, task, pkt):
        output_queue = task.get_outputs().get(0, None)
        # crop and send out
        ok, max_nms_area = self.process_ocr_results()
        self.write_result(ok, max_nms_area)
        if ok and output_queue:
            x0, y0, x1, y1 = max_nms_area
            x0 = int(x0 * self.frame_width)
            x1 = int(x1 * self.frame_width)
            y0 = int(y0 * self.frame_height)
            y1 = int(y1 * self.frame_height)
            crop_str = f"crop=w={x1-x0}:h={y1-y0}:x={x0}:y={y0}"
            Log.log(LogLevel.INFO, f"crop_str: {crop_str}")
            for packet in self.input_packets:
                vf = packet.get(VideoFrame)
                video_frame = ffmpeg.siso_filter(vf, crop_str)
                video_frame.pts = vf.pts
                video_frame.time_base = vf.time_base
                output_pkt = Packet(video_frame)
                output_pkt.timestamp = packet.timestamp
                output_queue.put(output_pkt)
        if output_queue:
            output_queue.put(pkt)
        task.timestamp = Timestamp.DONE
        self.reset()

    def on_pkt(self, task, pkt):
        if not pkt.is_(VideoFrame):
            return

        current_timestamp = pkt.timestamp * av_time_base
        if (
            self.last_processed_frame_timestamp is None
            or current_timestamp - self.last_processed_frame_timestamp >= 1
        ):
            vf = pkt.get(VideoFrame)
            if self.frame_width is None:
                self.frame_width = vf.width
                self.frame_height = vf.height
            frame = ffmpeg.reformat(vf, "rgb24").frame().plane(0).numpy()
            results = self.reader.readtext(frame)
            self.ocr_results.append(results)
            self.last_processed_frame_timestamp = pkt.timestamp * av_time_base

        self.input_packets.append(pkt)
