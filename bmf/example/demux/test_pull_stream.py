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


class TestVideoCModule(BaseTestCase):
    @timeout_decorator.timeout(seconds=120)
    def test_video(self):
        input_video_path = "../files/img_s.mp4"
        output_path = "./output.mp4"
        expect_result = '../c_module/output.mp4|1080|1920|7.615000|MOV,MP4,M4A,3GP,3G2,MJ2|4483410|4267646|h264|' \
                        '{"fps": "30.0662251656"}'
        self.remove_result_data(output_path)
        c_module_path = './libpull_stream.so'
        c_module_entry = 'pull_stream:PullStreamModule'
        # decode
        video_data_path = "/workspace/company/python2/bmf_c_engine_pure_test/bmf/video_content.txt"
        video_size_path = "/workspace/company/python2/bmf_c_engine_pure_test/bmf/video_length.txt"
        audio_data_path = "/workspace/company/python2/bmf_c_engine_pure_test/bmf/audio_content.txt"
        audio_size_path = "/workspace/company/python2/bmf_c_engine_pure_test/bmf/audio_length.txt"
        my_graph = bmf.graph()
        video = my_graph.module("pull_stream",
                                     {'input_path': input_video_path, 'entry': c_module_entry, 'data_path': video_data_path,
                                      "size_path": video_size_path,
                                      'module_path': c_module_path, 'module_type': "c++"}).decode()
        # audio = my_graph.module("pull_stream",
        #                            {'input_path': input_video_path, 'entry': c_module_entry, 'data_path': audio_data_path,
        #                             "size_path": audio_size_path,
        #                             'module_path': c_module_path, 'module_type': "c++"}).decode()
        # c module processing
        # video_2 = (
        #     video['video'].c_module("copy_module", c_module_path, c_module_entry)
        # )

        (
            bmf.encode(
                video["video"],  # video stream, set to None
                None,
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
