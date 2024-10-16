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
        graph = bmf.graph()
        scheduler_cnt = 1

        input_video_path = "../../../files/big_bunny_10s_30fps.mp4"
        output_path = "./output.mp4"
        expect_result = '../c_module/output.mp4|1080|1920|10.008|MOV,MP4,M4A,3GP,3G2,MJ2|1918880|2400520|h264|' \
                        '{"fps": "30.0662251656"}'
        self.remove_result_data(output_path)
        # decode
        video = graph.decode({'input_path': input_video_path})
        video.node_.scheduler_ = scheduler_cnt
        scheduler_cnt += 1
        # print(video["video"])
        # c module processing
        dist_nums = 3
        copymodule = bmf.module(
            [video['video']],
            "cpp_copy_module",
            option={
                "dist_nums": dist_nums,
            },
            module_path="./libcopy_module.so",
            entry="copy_module::CopyModule",
            input_manager="immediate",
            scheduler=scheduler_cnt
        )
        # copymodule.node_.scheduler_ = scheduler_cnt
        # video_2 = (video['video'].c_module("cpp_copy_module","./libcopy_module.so","copy_module::CopyModule").node_.schduler_ = schduler_cnt)
        scheduler_cnt += 1

        # encode
        encode = bmf.encode(
            copymodule['video'],  # video stream, set to None
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
            })
        
        encode.node_.scheduler_ = scheduler_cnt
        scheduler_cnt += 1
        graph.option_["scheduler_count"] = scheduler_cnt + dist_nums
        graph.run()
        self.check_video_diff(output_path, expect_result)


if __name__ == '__main__':
    unittest.main()
