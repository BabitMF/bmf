import sys
import time
import unittest

sys.path.append("../../..")
sys.path.append("../../c_module_sdk/build/bin/lib")
import bmf
from bmf import GraphConfig
import timeout_decorator

sys.path.append("../")
from base_test.base_test_case import BaseTestCase
from base_test.media_info import MediaInfo


class TestRunByConfig(BaseTestCase):
    @timeout_decorator.timeout(seconds=120)
    def test_run_by_config(self):
        input_video_path = "../files/img.mp4"
        output_path = "../files/out.mp4"
        expect_result = '../files/out.mp4|240|320|7.617000|MOV,MP4,M4A,3GP,3G2,MJ2|317620|302414|h264|' \
                        '{"fps": "30.0662251656"}'
        self.remove_result_data(output_path)
        # create graph
        my_graph = bmf.graph()
        file_path = 'config.json'
        # build GraphConfig instance by config file
        config = GraphConfig(file_path)

        # run
        my_graph.run_by_config(config)
        self.check_video_diff(output_path, expect_result)


if __name__ == '__main__':
    unittest.main()
