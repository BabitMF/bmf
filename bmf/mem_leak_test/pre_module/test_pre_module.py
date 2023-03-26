import sys
import time
import unittest

sys.path.append("../..")
import bmf
import timeout_decorator

sys.path.append("../")
from base_test.base_test_case import BaseTestCase
from base_test.media_info import MediaInfo


def create_node_test(module_name, option, input_video_path, output_path, pre_module=None):
    (
        bmf.graph({"dump_graph": 1})
            .decode({'input_path': input_video_path})['video']
            .scale(320, 240)
            .module(module_name, option, pre_module=pre_module)
            .encode(None, {
            "output_path": output_path,
            "video_params": {
                "width": 300,
                "height": 200,
            }
        }).run()
    )


class TestPreModule(BaseTestCase):
    def test_pre_module(self):
        input_video_path = "../files/img.mp4"
        output_path = "./output.mp4"

        # pre_allocate a module
        module_name = "analysis"
        option = {
            "name": "analysis_SR",
            "para": "analysis_SR"
        }
        pre_module = bmf.create_module(module_name, option)
        for i in range(3):
            create_node_test(module_name, option, input_video_path, output_path, pre_module=pre_module)


if __name__ == '__main__':
    unittest.main()
