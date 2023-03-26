import sys
import time
import unittest

sys.path.append("../../..")
import bmf
from bmf import bmf_sync, Packet

sys.path.append("../")
from base_test.base_test_case import BaseTestCase


class TestSyncMode(BaseTestCase):
    @timeout_decorator.timeout(seconds=120)
    def test_hw_encode(self):
        '''
        # for non-builtin modules, use module_info instead of module_name to specify type/path/entry
        
        module_info = {
            "name": "my_module",
            "type": "",
            "path": "",
            "entry": ""
        }
        module = bmf_sync.sync_module(module_info, {"input_path": input_video_path}, [], [0])
        '''
        c_module_path = './libhw_frame_gen.so'
        c_module_entry = 'hw_frame_gen:HwFrameGen'
        output_path = './test_hw_frame_encode.mp4'
        module_info = {
            "name": "hw_frame_gen",
            "type": "c++",
            "path": c_module_path,
            "entry": c_module_entry
        }
        hw_frame_gen = bmf_sync.sync_module(module_info, {"input_path": input_video_path}, [], [0])
        encoder = bmf_sync.sync_module("c_ffmpeg_encoder", {
            "output_path": output_path,
            "video_params": {
                "codec": "h264_nvenc"
            }
        }, [0], [])

        encoder.init()

        frames, _ = bmf_sync.process(hw_frame_gen, None)

        # encode
        bmf_sync.process(encoder, {0:frames[0]})

        bmf_sync.send_eof(encoder)

        encoder.close()


if __name__ == '__main__':
    unittest.main()
