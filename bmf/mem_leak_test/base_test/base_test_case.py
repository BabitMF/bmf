import sys
import time
import unittest
import os
import shutil
# import commands
import json
from .media_info import MediaInfo


class BaseTestCase(unittest.TestCase):
    def set_ffmpeg_env(self):
        ffmpeg_path = os.getenv('FFMPEG_BIN_PATH')
        if ffmpeg_path is not None:
            if ffmpeg_path not in os.environ["PATH"]:
                os.environ["PATH"] = os.environ["PATH"] + ":" + ffmpeg_path

    def remove_result_data(self, output):
        if os.path.exists(output):
            os.remove(output)

    def compare_info(self, output_path, encoded_video_info, expect_result):
        expected_result_list = expect_result.split('|')

        expected_height = int(expected_result_list[1])
        expected_width = int(expected_result_list[2])
        expected_duration = float(expected_result_list[3])
        expected_format = (str(expected_result_list[4])).upper()
        expected_bitrate = int(expected_result_list[5])
        expected_size = int(expected_result_list[6])
        expected_encoded_type = str(expected_result_list[7])
        expected_extra = expected_result_list[8]
        expected_extra = json.loads(expected_extra)
        expected_fps = float(expected_extra.get('fps', 0))
        encoded_height = int(encoded_video_info.get_height())
        encoded_width = int(encoded_video_info.get_width())
        encoded_duration = float(encoded_video_info.get_duration())
        encoded_format = (str(encoded_video_info.get_format()))
        encoded_bitrate = int(encoded_video_info.get_bitrate())
        encoded_size = int(encoded_video_info.get_size())
        encoded_encoded_type = str(encoded_video_info.get_encode_type())
        encoded_extra = encoded_video_info.get_extra_info()

        if encoded_height != expected_height or encoded_width != expected_width or encoded_format != expected_format \
                or encoded_encoded_type != expected_encoded_type:
            raise Exception('%s result not expected, one of height/width/format/encoded_type not the same,\
                            expected height:%s, width:%s, format:%s, encoded_type:%s, transcode height:%s, width:%s,\
                            format:%s, encoded_type:%s' % (
            output_path, expected_height, expected_width, expected_format,
            expected_encoded_type, encoded_height, encoded_width,
            encoded_format, encoded_encoded_type))
        if encoded_duration < expected_duration * (1 - 0.1) or encoded_duration > expected_duration * (1 + 0.1):
            raise Exception(
                "%s result not expected, duration not the same, encoded_duration:%s != expected_duration:%s" % (
                    output_path, encoded_duration, expected_duration))
        if encoded_bitrate < expected_bitrate * (1 - 0.2) or encoded_bitrate > expected_bitrate * (1 + 0.2):
            raise Exception(
                "%s result not expected, bitrate not the same, encoded_bitrate:%s != expected_bitrate:%s" % (
                    output_path, encoded_bitrate, expected_bitrate))
        if encoded_size < expected_size * (1 - 0.2) or encoded_size > expected_size * (1 + 0.2):
            raise Exception("%s result not expected, size not the same, encoded_size:%s != expected_size:%s" % (
                output_path, encoded_size, expected_size))
        if encoded_encoded_type == "":
            return
        if not encoded_extra:
            raise Exception("%s result not expected, extra is null" % (output_path))
        encoded_fps = float(encoded_extra.get('fps', 0))
        if encoded_fps < expected_fps * (1 - 0.1) or encoded_fps > expected_fps * (1 + 0.1):
            raise Exception("%s result not expected, fps not the same, encoded_fps:%s != expected_fps:%s" % (
                output_path, encoded_fps, expected_fps))

    def check_video_diff(self, output_path, expect_result):
        if os.path.exists(output_path):
            av_out_info = MediaInfo(output_path)
            self.compare_info(output_path, av_out_info, expect_result)
        else:
            raise Exception("output video :" + output_path + " not exist")
