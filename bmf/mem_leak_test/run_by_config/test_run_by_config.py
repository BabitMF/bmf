import sys
import unittest

sys.path.append("../../..")
sys.path.append("../../c_module_sdk/build/bin/lib")
import bmf
from bmf import GraphConfig
sys.path.append("../")
from base_test.base_test_case import BaseTestCase


class TestRunByConfig(BaseTestCase):
    def test_run_by_config(self):
        my_graph = bmf.graph()
        file_path = 'config.json'
        config = GraphConfig(file_path)
        my_graph.run_by_config(config)


if __name__ == '__main__':
    unittest.main()
