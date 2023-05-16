import sys
import os
import time
import unittest

sys.path.append("../../")
import bmf
import timeout_decorator

sys.path.append("../")
from base_test.base_test_case import BaseTestCase
from base_test.media_info import MediaInfo


class TestTranscode(BaseTestCase):
    @timeout_decorator.timeout(seconds=120)
    def test_set_option(self):
        input_video_path = "../files/img.mp4"
        input_video_path2 = "../files/single_frame.mp4"

        output_path = "./simple.mp4"
        # create graph
        graph = bmf.graph()

        # create graph
        graph = bmf.graph({'dump_graph': 1})

        # decode
        video = graph.decode({
            "input_path": input_video_path
        })['video']
        video2 = graph.decode({
            "input_path": input_video_path2
        })['video']

        vout = video.concat(video2)

        bmf.encode(
            vout,
            None,
            {
                "output_path": output_path,
                "video_params": {
                    "codec": "h264",
                    "width": 320,
                    "height": 240,
                    "crf": 23,
                    "preset": "veryfast"
                }
            }
        )

        graph_name = 'customed_name'
        graph.set_option({'graph_name': graph_name})

        graph_config, pre_module = graph.generate_graph_config()
        graph.dump_graph(graph_config)

        if not os.path.exists('original_' + graph_name + '.json'):
            raise Exception("customed graph file not exist")

if __name__ == '__main__':
    unittest.main()
