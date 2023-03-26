import sys
import time
import unittest

sys.path.append("../../..")
import bmf
import timeout_decorator

sys.path.append("../")
from base_test.base_test_case import BaseTestCase
from base_test.media_info import MediaInfo


class TestGenerator(BaseTestCase):
    def test_generator(self):
        pkts = (
            bmf.graph()
                .decode({'input_path': "../files/img.mp4"})['video']
                .ff_filter('scale', 299, 299)  # or you can use '.scale(299, 299)'
                .start()  # this will return a packet generator
        )

        for i, pkt in enumerate(pkts):
            # convert frame to a nd array
            if pkt.defined():
                frame = pkt.get(bmf.VideoFrame)
                np_frame = frame.to_image().image().data().numpy()

                # we can add some more processing here, e.g. predicting
                print('frame', i, 'shape', np_frame.shape)
            else:
                break


if __name__ == '__main__':
    unittest.main()
