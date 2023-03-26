import sys
import time
import unittest

sys.path.append("../../..")
import bmf
import timeout_decorator

sys.path.append("../")
from base_test.base_test_case import BaseTestCase
from base_test.media_info import MediaInfo


class TestAudioCModule(BaseTestCase):
    @timeout_decorator.timeout(seconds=120)
    def test_audio_c_module(self):
        input_video_path = "../files/img.mp4"
        output_path = "./audio_c_module.mp4"
        expect_result = 'audio_c_module.mp4|0|0|7.617000|MOV,MP4,M4A,3GP,3G2,MJ2|136031|129519||{}'
        self.remove_result_data(output_path)
        audio = bmf.graph().decode({'input_path': input_video_path})['audio'].module('my_module')
        bmf.encode(None, audio, {"output_path": output_path}).run()
        self.check_video_diff(output_path, expect_result)

    @timeout_decorator.timeout(seconds=120)
    def test_exception_in_python_module(self):
        input_video_path = "../files/img.mp4"
        output_path = "./test_exception_in_python_module.mp4"
        self.remove_result_data(output_path)
        audio = bmf.graph().decode({'input_path': input_video_path})['audio'].module('my_module', {"exception": 1})
        try:
            bmf.encode(None, audio, {"output_path": output_path}).run()
        except Exception as e:
            print(e)


if __name__ == '__main__':
    unittest.main()
