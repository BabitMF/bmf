import sys
import time
import unittest

sys.path.append("../../")
import bmf
import timeout_decorator

sys.path.append("../")
from base_test.base_test_case import BaseTestCase
from base_test.media_info import MediaInfo

class TestAVLog(BaseTestCase):
    def test_avlog(self):
        input_video_path = '../files/img.mp4'
        output_path = "./out.mp4"

        # create graph
        my_graph = bmf.graph()

        log_buff = my_graph.get_av_log_buffer()
        # otherwise log level can be set: log_buff = my_graph.get_av_log_buffer("debug")

        # decode video
        video1 = my_graph.decode({'input_path': input_video_path})

        # encode
        (
            bmf.encode(
                video1['video'],
                video1['audio'],
                {
                    "video_params": {
                        "codec": "h264",
                        "width": 320,
                        "height": 240,
                        "crf": 23,
                        "preset": "veryfast",
                        "x264-params": "ssim=1:psnr=1"
                    },
                    "output_path": output_path
                }
            ).run()
        )

        file_name = "avlog.txt"
        f = open(file_name, 'w')
        f.write(''.join(log_buff))
        f.close()

        found = 0
        check_file = open(file_name, 'r')
        for line in check_file.readlines():
            if line.find('SSIM Mean') > -1:
                found += 1
            if line.find('PSNR Mean') > -1:
                found += 1
        check_file.close()

        self.assertEqual(found, 5)

if __name__ == '__main__':
    unittest.main()
