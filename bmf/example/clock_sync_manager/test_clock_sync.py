import sys
import unittest

sys.path.append("../../..")
sys.path.append("../../c_module_sdk/build/bin/lib")
import bmf

# import timeout_decorator

sys.path.append("../")
from base_test.base_test_case import BaseTestCase


class TestClockSyncModule(BaseTestCase):
    def test_video(self):
        input_video_path = "../files/img_s.mp4"
        output_path = "./output.mp4"
        expect_result = '../c_module/output.mp4|1080|1920|7.615000|MOV,MP4,M4A,3GP,3G2,MJ2|4483410|4267646|h264|' \
                        '{"fps": "30.0662251656"}'
        self.remove_result_data(output_path)
        # decode/Users/bytedance/Project/company/python2/bmf_c_engine_pure_test/bmf/bmf/example/audio_decode
        video_data_path = "/workspace/company/python2/bmf_c_engine_pure_test/bmf/video_content.txt"
        video_size_path = "/workspace/company/python2/bmf_c_engine_pure_test/bmf/video_length.txt"

        graph = bmf.graph()

        video = graph.decode({
            "input_path": input_video_path
        })
        video["audio"].upload()
        clock = graph.c_module("Clock", option={"fps": 60})
        v = bmf.module([clock, video["video"]], "video_layout",option={
            "alias": "layout",
            "layout_mode": "gallery",
            "crop_mode": "",
            "layout_location": "",
            "interspace": 0,
            "main_stream_idx": 0,
            "width": 1920,
            "height": 1080,
            "background_color": "#123456"
        }, input_manager="clocksync")

        (
            # video.run()
            bmf.encode(
                # out_video,
                # None,
                v,  # video stream, set to None
                # audio1,
                None,
                # {"output_path": output_path,"audio_params":{"sample_rate":44100,"codec": "aac"}}
                {"output_path": output_path}
            ).run()
        )

        # (
        #     bmf.encode(
        #         None,  # video stream, set to None
        #         audio["audio"],
        #         {"output_path": output_path}
        #     ).run()
        # )

        # encode
        # (
        #     bmf.encode(
        #         video["video"],  # video stream, set to None
        #         audio["audio"],
        #         {"output_path": output_path}
        #     ).run()
        # )
        # self.check_video_diff(output_path, expect_result)


if __name__ == '__main__':
    unittest.main()
