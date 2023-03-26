#!/usr/bin/env python
# -*- coding: utf-8 -*-
import json
import subprocess

FFPROBE = 'ffprobe'


def ff_probe(input_url, extra_options=""):
    show_options = " -show_format -show_streams " + extra_options
    ff_cmd = "%s -hide_banner -loglevel quiet -print_format json %s %s" % (FFPROBE, show_options, input_url)
    (status, raw_output) = subprocess.getstatusoutput(ff_cmd)
    if status:
        raise Exception('may be problem video, status_code %s != 0' % (status))
    dict_info = json.loads(raw_output)
    if 'format' not in dict_info:
        raise Exception('no format in output')
    if 'streams' not in dict_info or len(dict_info['streams']) <= 0:
        raise Exception('ffprobe no streams')
    av_out_info = dict_info
    for stream in dict_info['streams']:
        if stream.get('codec_type') == 'video' and 'v_stream' not in av_out_info:
            av_out_info['v_stream'] = stream
        elif stream.get('codec_type') == 'audio' and 'a_stream' not in av_out_info:
            av_out_info['a_stream'] = stream
    return av_out_info


def get_video_duration(ffprobe_result):
    v_stream = ffprobe_result.get('v_stream', {})
    # audio
    if not v_stream:
        return -1
    duration = float(v_stream.get('duration', "-1"))
    if duration == -1:
        format = ffprobe_result.get('format', {})
        duration = float(format.get('duration', "-1"))
    # image
    if duration <= 0:
        return 0
    return duration


def get_audio_duration(ffprobe_result):
    a_stream = ffprobe_result.get('a_stream', {})
    # video or image
    if not a_stream:
        return -1
    duration = float(a_stream.get('duration', "-1"))
    if duration == -1:
        format = ffprobe_result.get('format', {})
        duration = float(format.get('duration', "-1"))
    if duration <= 0:
        raise Exception('invalid audio duration')
    return duration


def get_filetype(ffprobe_result):
    format_name = ffprobe_result["format"].get('format_name')
    if format_name.startswith("mov"):
        filetype = "mov"
    elif format_name.startswith("png"):
        filetype = "png"
    elif format_name.startswith("matroska,webm"):
        filetype = "webm"
    elif format_name.startswith("gif"):
        filetype = "gif"
    else:
        filetype = format_name
    return filetype


def get_video_size(ffprobe_result):
    v_stream = ffprobe_result.get('v_stream', {})
    # audio
    if not v_stream:
        return -1, -1
    width = float(v_stream.get('width', "-1"))
    height = float(v_stream.get('height', "-1"))
    if width == -1 or height == -1:
        raise Exception('invalid height or width')
    return width, height
