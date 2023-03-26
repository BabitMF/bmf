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


class TestNormalCModule(BaseTestCase):
    @timeout_decorator.timeout(seconds=120)
    def test_normal_c_module(self):
        c_module_path = './'
        c_module_name = 'normal_image_info:NormalImageInfo'
        graph = bmf.graph({"dump_graph": 1})

        # decode
        produce_image_info = graph.module("produce_image_info", {"num": 100})

        (
            produce_image_info.c_module(c_module_path, c_module_name).upload().run()
        )


if __name__ == '__main__':
    unittest.main()
