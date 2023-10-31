import sys
import time
import unittest

sys.path.append("../..")
# sys.path.append("../../c_module_sdk/build/bin/lib")
import bmf
import os
if os.name == 'nt':
    # We redefine timeout_decorator on windows
    class timeout_decorator:

        @staticmethod
        def timeout(*args, **kwargs):
            return lambda f: f  # return a no-op decorator
else:
    import timeout_decorator

sys.path.append("../../test/")
from base_test.base_test_case import BaseTestCase
from base_test.media_info import MediaInfo

sys.path.append("./c_module")


class TestVideoCModule(BaseTestCase):

    @timeout_decorator.timeout(seconds=120)
    def test_video(self):
        input_video_path = "../../files/big_bunny_10s_30fps.mp4"
        output_path = "./output.mp4"
        expect_result = '../c_module/output.mp4|1080|1920|10.008|MOV,MP4,M4A,3GP,3G2,MJ2|1918880|2400520|h264|' \
                        '{"fps": "30.0662251656"}'
        self.remove_result_data(output_path)
        # decode
        video = bmf.graph().decode({'input_path': input_video_path})
        # c module processing
        video_2 = (video['video'].c_module("cpp_copy_module"))

        # encode
        (bmf.encode(
            video_2,  # video stream, set to None
            video['audio'],
            {
                "output_path": output_path,
                "video_params": {
                    "vsync": "vfr",
                    "max_fr": 60
                },
                "audio_params": {
                    "codec": "aac"
                }
            }).run())
        self.check_video_diff(output_path, expect_result)

    @timeout_decorator.timeout(seconds=120)
    def test_image(self):
        input_video_path = "../../files/overlay.png"
        output_path = "output.jpg"
        expect_result = 'output.jpg|240|320|0.040000|IMAGE2|975400|4877|mjpeg|{"fps": "25.0"}'
        self.remove_result_data(output_path)
        (bmf.graph().decode({'input_path': input_video_path})['video'].scale(
            320, 240).c_module("cpp_copy_module").encode(
                None, {
                    "output_path": output_path,
                    "format": "mjpeg",
                    "video_params": {
                        "codec": "jpg",
                        "width": 320,
                        "height": 240
                    }
                }).run())
        self.check_video_diff(output_path, expect_result)


if __name__ == '__main__':
    unittest.main()
