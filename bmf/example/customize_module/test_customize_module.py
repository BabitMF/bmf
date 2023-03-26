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


class TestCustomizeModule(BaseTestCase):
    @timeout_decorator.timeout(seconds=120)
    def test_customize_module(self):
        input_video_path = "../files/img.mp4"
        output_path = "./output.mp4"
        expect_result = '|1080|1920|7.615000|MOV,MP4,M4A,3GP,3G2,MJ2|4483427|4267663|h264|' \
                        '{"fps": "30.0662251656"}'
        self.remove_result_data(output_path)
        (
            bmf.graph()
                .decode({'input_path': input_video_path})['video']
                .module('my_module')
                .encode(None, {"output_path": output_path})
                .run()
        )
        self.check_video_diff(output_path, expect_result)


if __name__ == '__main__':
    unittest.main()
