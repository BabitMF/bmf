import sys
import time
import unittest

sys.path.append("../../..")
sys.path.append("../../c_module_sdk/build/bin/lib")
import bmf
from bmf import VideoFrame
import timeout_decorator

sys.path.append("../")
from base_test.base_test_case import BaseTestCase
from base_test.media_info import MediaInfo


class TestAudioCModule(BaseTestCase):
    @timeout_decorator.timeout(seconds=120)
    def test_audio_c_module(self):
        input_video_path = "../files/img.mp4"
        output_path = "./audio.mp4"
        expect_result = '|1080|1920|7.615000|MOV,MP4,M4A,3GP,3G2,MJ2|4483427|4267663|h264|' \
                        '{"fps": "30.0662251656"}'
        self.remove_result_data(output_path)
        video = bmf.graph().decode({'input_path': input_video_path})
        c_module_path = './'
        c_module_name = 'copy_module:CopyModule'
        # c module processing
        audio_2 = (
            video['audio'].c_module(c_module_path, c_module_name)
        )
        # encode
        (
            bmf.encode(
                video['video'],  # video stream, set to None
                audio_2,
                {"output_path": output_path, "audio_params": {"codec": "aac"}}
            ).run()
        )
        self.check_video_diff(output_path, expect_result)


def test():
    input_video_path = "../files/img.mp4"
    # input_video_path = "audio.mp4"
    media_info = MediaInfo(input_video_path)
    # print(media_info.av_out_info)
    print(media_info.trans2expect_value())


if __name__ == '__main__':
    unittest.main()
