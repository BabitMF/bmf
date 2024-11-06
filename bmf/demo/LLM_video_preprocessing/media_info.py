#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import subprocess
import json

logger = logging.getLogger('main')

class MediaInfo:
    def __init__(self, ffprobe_bin, input_file):
        self.__ffprobe_bin = ffprobe_bin
        self.__streams = dict()
        self.__v_stream = {}
        self.__v_stream_idx = 0
        self.__a_stream = {}
        self.__a_stream_idx = 0
        show_options = " -show_format -show_streams"
        ff_cmd = "%s -hide_banner -loglevel error -print_format json %s %s" % (
            self.__ffprobe_bin,
            show_options,
            input_file,
        )
        res = subprocess.Popen(ff_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = res.communicate()
        status = res.returncode
        if status:
            raise Exception('ffprobe failed')
        msg = stdout.decode(errors='ignore')
        dict_info = json.loads(msg)
        if 'streams' not in dict_info or len(dict_info['streams']) <= 0:
            raise Exception('ffprobe no streams')
        streams = dict_info.get('streams')
        self.__streams = streams
        self.__auto_select_stream()

    def __auto_select_stream(self):
        v_streams = []
        a_streams = []
        streams = self.__streams
        for stream in streams:
            if stream.get('codec_type') == 'video':
                v_streams.append(stream)
            elif stream.get('codec_type') == 'audio':
                a_streams.append(stream)
        if len(v_streams) > 0:
            self.__v_stream = v_streams[0]
        if len(a_streams) > 0:
            self.__a_stream = a_streams[0]

    def video_wxh(self):
        v_stream = self.__v_stream
        src_w = int(v_stream.get('width', '-1'))
        src_h = int(v_stream.get('height', '-1'))
        if src_w < 0 or src_h < 0:
            raise Exception('invalid resolution')
        return src_w, src_h

    def video_duration(self):
        v_stream = self.__v_stream
        video_duration = -1
        if 'duration' in v_stream:
            video_duration = float(v_stream.get('duration', -1))
        return video_duration



