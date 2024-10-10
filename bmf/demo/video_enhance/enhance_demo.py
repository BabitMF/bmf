#!/usr/bin/env python
# -*- coding: utf-8 -*-

input_file = "../../files/big_bunny_10s_30fps.mp4"
output_file = "output.mp4"
output_compose_file = "output_compose.mp4"

import bmf


def run():
    graph = bmf.graph()
    video = graph.decode({
        "input_path": input_file,
    })

    enhance = bmf.module(
        [video["video"]],
        "EnhanceModule",
        option={
            "fp32": True,
            "output_scale": 2,
            "thread": 3
        },
        entry="enhance_module.EnhanceModule",
        input_manager="immediate",
    )

    origin_video_scale = video["video"].scale("2*in_w:2*in_h")

    com = bmf.module(
        [enhance[0], origin_video_scale],
        "CompositionModule",
        entry="composition_module.CompositionModule",
        input_manager="framesync",
    )

    bmf.encode(
        enhance[0],
        video["audio"],
        {
            "video_params": {
                "g": "50",
                "preset": "veryfast",
                "bf": 0,
                "vsync": "vfr",
                "max_fr": 30,
            },
            "audio_params": {
                "sample_rate": 44100,
                "codec": "aac"
            },
            "loglevel": "info",
            "output_path": output_file,
        },
    )

    bmf.encode(
        com[0],
        video["audio"],
        {
            "video_params": {
                "g": "50",
                "preset": "veryfast",
                "bf": 0,
                "vsync": "vfr",
                "max_fr": 30,
            },
            "audio_params": {
                "sample_rate": 44100,
                "codec": "aac"
            },
            "loglevel": "info",
            "output_path": output_compose_file,
        },
    )
    graph.run()


if __name__ == "__main__":
    run()
