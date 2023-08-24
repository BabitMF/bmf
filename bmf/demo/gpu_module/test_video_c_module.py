import sys
import time
import unittest

sys.path.append("../..")
# sys.path.append("../../c_module_sdk/build/bin/lib")
import bmf
import timeout_decorator

sys.path.append("../../test/")
from base_test.base_test_case import BaseTestCase
from base_test.media_info import MediaInfo

sys.path.append("./c_module")


class TestVideoCModule(BaseTestCase):

    @timeout_decorator.timeout(seconds=120)
    def test_video(self):
        input_video_path = "../../files/img.mp4"
        output_path = "./output.mp4"
        expect_result = '../c_module/output.mp4|1080|1920|7.615000|MOV,MP4,M4A,3GP,3G2,MJ2|4483410|4267646|h264|' \
                        '{"fps": "30.0662251656"}'
        self.remove_result_data(output_path)
        c_module_path = './libcvtcolor.so'
        c_module_entry = 'cvtcolor:TestCvtColorModule'
        # decode
        video = bmf.graph().decode({
            'input_path': input_video_path,
            'video_params': {
                'pix_fmt': 'nv12'
            }
        })
        # c module processing
        video_2 = (video['video'].ff_filter('format', 'yuv420p').c_module(
            'cvtcolor', c_module_path, c_module_entry))
        # ffmpeg filter
        # video_3 = (
        #     video_2['video'].ff_filter('format', 'nv12')
        # )

        # encode
        # video_3.encode(
        #     None,
        #     {"output_path": output_path,
        #      "video_params": {
        #         "vsync": "vfr",
        #         "max_fr": 60
        #      },
        #       "audio_params": {"codec": "aac"}
        #     }
        # )
        (bmf.encode(
            video_2,  # video stream, set to None
            video['audio'],
            {
                "output_path": output_path,
                "video_params": {
                    "codec": "hevc_nvenc",
                    "width": 1920,
                    "height": 1080,
                    #  "vsync": "vfr",
                    #  "max_fr": 60
                },
                "audio_params": {
                    "codec": "aac"
                }
            }).run())
        # self.check_video_diff(output_path, expect_result)

    # @timeout_decorator.timeout(seconds=120)
    # def test_image(self):
    #     input_video_path = "../../files/overlay.png"
    #     output_path = "output.jpg"
    #     expect_result = 'output.jpg|240|320|0.040000|IMAGE2|975400|4877|mjpeg|{"fps": "0"}'
    #     self.remove_result_data(output_path)
    #     c_module_path = './libcvtcolor.so'
    #     c_module_entry = 'cvtcolor:TestCvtColorModule'
    #     (
    #         bmf.graph()
    #             .decode({'input_path': input_video_path})['video']
    #             .scale(320, 240)
    #             .c_module("cvtcolor", c_module_path, c_module_entry)
    #             .encode(None, {
    #             "output_path": output_path,
    #             "format": "mjpeg",
    #             "video_params": {
    #                     "codec": "jpg",
    #                     "width": 320,
    #                     "height": 240
    #             }

    #         }).run()
    #     )
    # self.check_video_diff(output_path, expect_result)


if __name__ == '__main__':
    unittest.main()
