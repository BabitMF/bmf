#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import json
import sys
import string

if sys.version_info.major == 2:
    from commands import getstatusoutput
else:
    from subprocess import getstatusoutput


class MediaInfo(object):
    def __init__(self, video_path):
        self.video_path_ = video_path
        ffmpeg_path = os.getenv('FFMPEG_BIN_PATH')
        if ffmpeg_path is None:
            FFPROBE = 'ffprobe'
        else:
            FFPROBE = ffmpeg_path + "/ffprobe"
        show_options = " -show_format -show_streams "
        ff_cmd = "%s -hide_banner -loglevel quiet -print_format json %s %s" % (FFPROBE, show_options, video_path)
        (status, raw_output) = getstatusoutput(ff_cmd)
        if status:
            raise Exception('may be problem video, status_code %s != 0' % (status))
        dict_info = json.loads(raw_output)
        if 'format' not in dict_info:
            raise Exception('no format in output')
        if 'streams' not in dict_info or len(dict_info['streams']) <= 0:
            raise Exception('ffprobe no streams')
        self.av_out_info = dict_info
        for stream in dict_info['streams']:
            if stream.get('codec_type') == 'video' and 'v_stream' not in self.av_out_info:
                self.av_out_info['v_stream'] = stream
            elif stream.get('codec_type') == 'audio' and 'a_stream' not in self.av_out_info:
                self.av_out_info['a_stream'] = stream

    def get_width(self, default_value=0):
        if 'v_stream' in self.av_out_info:
            return int(self.av_out_info['v_stream'].get("width", default_value))
        else:
            return default_value

    def get_height(self, default_value=0):
        if 'v_stream' in self.av_out_info:
            return int(self.av_out_info['v_stream'].get("height", default_value))
        else:
            return default_value

    def get_duration(self, default_value=0):
        duration = float(self.av_out_info["format"].get('duration', default_value))
        return duration

    def get_format(self, default_value=""):
        format_name = (self.av_out_info["format"].get('format_name', default_value)).upper()
        return format_name

    def get_bitrate(self, default_value=0):
        bit_rate = int(self.av_out_info["format"].get('bit_rate', default_value))
        return bit_rate

    def get_size(self, default_value=0):
        size = int(self.av_out_info["format"].get('size', default_value))
        return size

    def get_encode_type(self, default_value=""):
        if 'v_stream' in self.av_out_info:
            return self.av_out_info['v_stream'].get("codec_name", default_value)
        else:
            return default_value

    def parse_fraction(self, fraction_str):
        num_list = fraction_str.split('/')
        if len(num_list) == 1:
            return float(num_list[0])
        if len(num_list) > 2:
            raise Exception('invalid fraction')
        if float(num_list[1]) == 0:
            return 0
        return float(num_list[0]) / float(num_list[1])

    def get_extra_info(self, default_value=None):
        if default_value is None:
            default_value = {}
        result = dict()
        if 'v_stream' in self.av_out_info:
            fps_str = self.av_out_info.get('v_stream').get('avg_frame_rate', '-1/1')
            fps = self.parse_fraction(fps_str)
            result = {'fps': str(fps)}
            return result
        else:
            return {}

    def trans2expect_value(self):
        encoded_height = int(self.get_height())
        encoded_width = int(self.get_width())
        encoded_duration = float(self.get_duration())
        encoded_format = (str(self.get_format()))
        encoded_bitrate = int(self.get_bitrate())
        encoded_size = int(self.get_size())
        encoded_encoded_type = str(self.get_encode_type())
        encoded_extra = self.get_extra_info()
        result = "%s|%d|%d|%f|%s|%d|%d|%s|%s" % (self.video_path_, encoded_height, encoded_width, encoded_duration,
                                                 encoded_format, encoded_bitrate, encoded_size, encoded_encoded_type,
                                                 str(encoded_extra))
        return result


def test(argv):
    input_video_path = argv[1]
    # input_video_path = "audio.mp4"
    media_info = MediaInfo(input_video_path)
    print(media_info.av_out_info)
    print(media_info.trans2expect_value().replace("'", '"'))


if __name__ == '__main__':
    import sys

    test(sys.argv)
