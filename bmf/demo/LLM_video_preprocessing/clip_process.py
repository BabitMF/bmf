#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging
import bmf
from pydantic.v1.utils import deep_update
import os
import json
from pathlib import Path

logger = logging.getLogger("main")

DEFAULT_OPERATOR_CONFIG = {
    "ocr_crop": {
        "name": "ocr_crop",
        "module_name": "ocr_crop",
        "pre_module": True,
        "params": {},
    },
    "aesmod_module": {
        "name": "aesmod_module",
        "module_name": "aesmod_module",
        "pre_module": False,
        "params": {}
    },
}

ClipOutputPath = "clip_output"

FIXED_RES = 480

def prepare_dir(base_path):
    Path(base_path).mkdir(parents=True, exist_ok=True)

def get_operator_tmp_result_path(base_path, operator_name):
    file_name = f"{operator_name}_result.json"
    return os.path.join(base_path, file_name)


def get_operator_result_path(base_path, operator_name, clip_index):
    file_name = f"clip_{clip_index}_{operator_name}_result.json"
    return os.path.join(base_path, file_name)


def get_jpg_serial_path(base_path, clip_index, fixed_width):
    img_path = f"clip_{clip_index}_{fixed_width}_img"
    return os.path.join(
        base_path, img_path, "clip_{}_{}_img_%04d.jpg".format(clip_index, fixed_width)
    )

def get_jpg_dir_path(base_path, clip_index, fixed_width):
    img_path = f"clip_{clip_index}_{fixed_width}_img"
    return os.path.join(base_path, img_path)

def get_clip_path(base_path, clip_index, fixed_width):
    file_name = "clip_{}_{}.mp4".format(clip_index, fixed_width)
    return os.path.join(base_path, file_name)


def bmf_output(stream, configs, clip_index):
    if len(configs) == 0:
        return
    s = stream
    resize_streams = dict()

    for c in configs:
        o = s
        res_str = c["res"]
        if res_str != "orig":
            stream_key = res_str
            if stream_key not in resize_streams:
                res = int(res_str)
                o = o.scale(
                    "if(lt(iw,ih),{},-2):if(lt(iw,ih),-2,{})".format(res, res)
                ).ff_filter("setsar", sar="1/1")
                resize_streams[stream_key] = o
            else:
                # get saved resized stream
                o = resize_streams[stream_key]

        elif "limit" in c:
            res = int(c["limit"])
            o = o.scale(
                f"if(lt(iw,ih),min({res},iw),-2):if(lt(iw,ih),-2,min({res},ih))"
            ).ff_filter("setsar", sar="1/1")

        # encode
        if c["type"] == "jpg":
            bmf.encode(
                o,
                None,
                {
                    "output_path": get_jpg_serial_path(
                        ClipOutputPath, clip_index, c["res"]
                    ),
                    "video_params": {
                        "vsync": "vfr",
                        "codec": "jpg",
                        "qscale": int(c["quality"]) if "quality" in c else 2,
                        "pix_fmt": "yuvj444p",
                        "threads": "4",
                    },
                    "format": "image2",
                },
            )

        elif c["type"] == "mp4":
            bmf.encode(
                o,
                None,
                {
                    "output_path": get_clip_path(ClipOutputPath, clip_index, c["res"]),
                    "video_params": {
                        "vsync": "vfr",
                        "codec": "h264",
                        "preset": "veryfast",
                        "threads": "4",
                    },
                },
            )


class ClipProcess:
    def __init__(self, input_file, timelines, mode, config):
        self.input_file = input_file
        self.timelines = timelines
        self.mode = mode
        self.config = config
        modes = mode.split(",")
        operator_options = []
        self.output_configs = (
            config["output_configs"] if "output_configs" in config else {}
        )
        for operator_name in modes:
            if operator_name in DEFAULT_OPERATOR_CONFIG:
                option = DEFAULT_OPERATOR_CONFIG[operator_name]
                config_operator_option = (
                    config[operator_name] if operator_name in config else {}
                )
                operator_option = deep_update(option, config_operator_option)
                operator_options.append(operator_option)
        logger.info(f"operator_options: {operator_options}")
        self.operator_options = operator_options
        prepare_dir(ClipOutputPath)

    def operator_process(self, timeline):
        if len(self.operator_options) == 0:
            return True

        decode_param = {}
        decode_param["input_path"] = self.input_file
        decode_param["dec_params"] = {"threads": "4"}
        decode_param["durations"] = timeline
        graph = bmf.graph()
        v = graph.decode(decode_param)["video"]
        v = v.scale(
            f"if(lt(iw,ih),min({FIXED_RES},iw),-2):if(lt(iw,ih),-2,min({FIXED_RES},ih))"
        ).ff_filter("setsar", sar="1/1")
        for operator_option in self.operator_options:
            operator_name = operator_option["name"]
            if "module_name" in operator_option:
                if operator_option["pre_module"]:
                    v = v.module(
                        operator_option["module_name"],
                        pre_module=self.operator_premodules[operator_name],
                    )
                else:
                    v = v.module(
                        operator_option["module_name"],
                        option={
                            "result_path": get_operator_tmp_result_path(
                                ClipOutputPath, operator_name
                            ),
                            "params": operator_option["params"],
                        },
                    )
        pkts = v.start()
        count = 0
        for _, pkt in enumerate(pkts):
            if pkt.is_(bmf.VideoFrame):
                count += 1
        logger.info(f"operator process get videoframe count: {count}")
        return count > 0

    def process_one_clip(self, timeline, clip_index):
        passthrough = self.operator_process(timeline)

        if not passthrough:
            return

        if len(self.output_configs) == 0:
            return

        for output in self.output_configs:
            if output["type"] == "jpg":
                res = output["res"]
                img_dir = get_jpg_dir_path(ClipOutputPath, clip_index, res)
                prepare_dir(img_dir)

        decode_param = {}
        decode_param["input_path"] = self.input_file
        decode_param["dec_params"] = {"threads": "4"}
        decode_param["durations"] = timeline
        graph = bmf.graph({"optimize_graph": False})
        v = graph.decode(decode_param)["video"]

        for operator_option in self.operator_options:
            operator_name = operator_option["name"]
            if operator_name == "ocr_crop":
                result_path = get_operator_tmp_result_path(
                    ClipOutputPath, operator_name
                )
                if not os.path.exists(result_path):
                    continue
                with open(result_path, "r") as f:
                    operator_res = json.load(f)
                if (
                    "result" in operator_res
                    and "nms_crop_box" in operator_res["result"]
                ):
                    nms_crop_box = operator_res["result"]["nms_crop_box"]
                    left, top, right, bottom = nms_crop_box
                    v = v.ff_filter(
                        "crop",
                        f"w=iw*{right - left}:h=ih*{bottom - top}:x=iw*{left}:y=ih*{top}",
                    )

        bmf_output(v, self.output_configs, clip_index)
        graph.run()

    def process_clip_result(self, clip_index):
        for operator_option in self.operator_options:
            operator_name = operator_option["name"]
            tmp_path = get_operator_tmp_result_path(ClipOutputPath, operator_name)
            if os.path.exists(tmp_path):
                os.rename(
                    tmp_path,
                    get_operator_result_path(ClipOutputPath, operator_name, clip_index),
                )

    def process(self):
        # create premodule
        operator_premodules = {}
        for op_option in self.operator_options:
            if "pre_module" in op_option and op_option["pre_module"]:
                operator_name = op_option["name"]
                operator_premodule = bmf.create_module(
                    op_option["module_name"],
                    option={
                        "result_path": get_operator_tmp_result_path(
                            ClipOutputPath, operator_name
                        ),
                        "params": op_option["params"],
                    },
                )
                operator_premodules[operator_name] = operator_premodule

        self.operator_premodules = operator_premodules

        for i in range(len(self.timelines)):
            timeline = self.timelines[i]
            self.process_one_clip(timeline, i)
            self.process_clip_result(i)
