#!/usr/bin/env python
# -*- coding: utf-8 -*-

import bmf

def get_split_info(video_path, scene_thres):
    graph = bmf.graph()
    v = graph.decode({"input_path":video_path})['video']
    v = v.ff_filter('select', f"gt(scene,{scene_thres})").pass_through()
    pkts = v.start()
    scene_change_list = []
    for _, pkt in enumerate(pkts):
        if pkt.is_(bmf.VideoFrame):
            scene_change_list.append(pkt.timestamp)
    return scene_change_list

#print(get_split_info("/home/huheng.1989/Videos/hdr_720p.mp4", 0.3))
