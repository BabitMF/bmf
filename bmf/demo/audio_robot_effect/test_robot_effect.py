import bmf
import os
import unittest
import sys

if os.name == 'nt':
    # We redefine timeout_decorator on windows
    class timeout_decorator:

        @staticmethod
        def timeout(*args, **kwargs):
            return lambda f: f  # return a no-op decorator
else:
    import timeout_decorator

sys.path.append("../..")
sys.path.append("../../test/")
from base_test.base_test_case import BaseTestCase
from base_test.media_info import MediaInfo

class TestVideoCModule(BaseTestCase):

    @timeout_decorator.timeout(seconds=120)
    def test_robot_effect(self):

        root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        module_path = os.path.join(root_path, 'modules')

        py_module_info = {
            "name": "audio_robot_effect_module",
            "type": "",
            "path": module_path,
            "entry": "robot_effect:BMFRobotEffect"
        }   

        input_path = '../../files/counting_number.wav'
        output_path = './robot_effect.wav'

        streams = bmf.graph().decode({'input_path': input_path})
        audio_stream = streams['audio'].module(py_module_info, None)

        (bmf.encode(
            None,
            audio_stream,
            {
                "output_path": output_path,
                "audio_params": {
                    "codec": "aac",
                    "bit_rate": 128000,
                    "sample_rate": 44100,
                    "channels": 2
                }
            }
        ).run())

if __name__ == "__main__":
    unittest.main()

