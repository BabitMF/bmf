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


class TestSubgraph(BaseTestCase):
    @timeout_decorator.timeout(seconds=120)
    def test_subgraph(self):
        input_video_path = "../files/img.mp4"
        input_over_lay_image = "../files/overlay.png"
        output_path = "./output.mp4"
        expect_result = '../subgraph/output.mp4|1080|1920|7.615000|MOV,MP4,M4A,3GP,3G2,MJ2|5571174|5303062|h264|' \
                        '{"fps": "30.0662251656"}'
        self.remove_result_data(output_path)
        # create graph
        graph = bmf.graph()

        # decode video
        video = graph.decode({'input_path': input_video_path})

        # decoder overlay image
        overlay = graph.decode({'input_path': input_over_lay_image})

        # call sub graph and encoder
        (
            bmf.module([video['video'], overlay['video']], 'subgraph_module')
                .encode(video['audio'], {"output_path": output_path})
                .run()
        )
        self.check_video_diff(output_path, expect_result)


if __name__ == '__main__':
    unittest.main()
