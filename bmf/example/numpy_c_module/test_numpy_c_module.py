import sys
import time
import unittest

sys.path.append("../../..")
sys.path.append("../../c_module_sdk/build/bin/lib")
import bmf
import timeout_decorator

sys.path.append("../")
from base_test.base_test_case import BaseTestCase
from base_test.media_info import MediaInfo


class TestNumpyCModule(BaseTestCase):
    @timeout_decorator.timeout(seconds=120)
    def test_video(self):
        input_video_path = "../files/img.mp4"
        output_path = "./output.mp4"
        expect_result = '../c_module/output.mp4|1080|1920|7.615000|MOV,MP4,M4A,3GP,3G2,MJ2|4483410|4267646|h264|' \
                        '{"fps": "30.0662251656"}'
        self.remove_result_data(output_path)
        c_module_path = './'
        c_module_name = 'copy_module:CopyModule'
        # decode
        video = bmf.graph().decode({'input_path': input_video_path})

        # c module processing
        video_2 = (
            video['video'].module("type_conversion", {"to_numpy": 1}).c_module(c_module_path, c_module_name).module(
                "type_conversion", {"to_numpy": 0})
        )

        # encode
        (
            bmf.encode(
                video_2,  # video stream, set to None
                None,
                {"output_path": output_path,
                 "audio_params": {"codec": "aac"}
                 }
            ).run()
        )
        self.check_video_diff(output_path, expect_result)


if __name__ == '__main__':
    unittest.main()
