#!/usr/bin/env python
# -*- coding: utf-8 -*-
import bmf


def test():
    graph = bmf.graph({'dump_graph':1})
    video = graph.decode({"input_path": "../../files/big_bunny_10s_30fps.mp4"})[
        "video"
    ]
    vo = video.fps(24).pass_through()

    v = vo.scale(848,480).setsar(1).pass_through()

    s = bmf.module(
        [vo, v],
        "switch_module",
        input_manager="immediate",
    )

    v1 = s.scale(480,320)
    v2 = v.scale(640, 360)
    v3 = v.scale(360, 240)

    bmf.encode(
        v1,
        None,
        {
            "output_path": "v1.mp4",
            "video_params": {
                "vsync": "vfr",
                "codec": "h264",
                "preset": "medium",
            },
        },
    )

    bmf.encode(
        v2,
        None,
        {
            "output_path": "v2.mp4",
            "video_params": {
                "vsync": "vfr",
                "codec": "h264",
                "preset": "medium",
            },
        },
    )

    bmf.encode(
        v3,
        None,
        {
            "output_path": "v3.mp4",
            "video_params": {
                "vsync": "vfr",
                "codec": "h264",
                "preset": "medium",
            },
        },
    )
    graph.run()


if __name__ == "__main__":
    test()
