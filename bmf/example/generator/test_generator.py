import sys
import time
import unittest

sys.path.append("../../..")
import bmf
import bmf.hml.hmp as mp
import timeout_decorator


sys.path.append("../")
from base_test.base_test_case import BaseTestCase
from base_test.media_info import MediaInfo


class TestGenerator(BaseTestCase):
    @timeout_decorator.timeout(seconds=120)
    def test_generator(self):
        pkts = (
            bmf.graph()
            .decode({"input_path": "../files/img.mp4"})["video"]
            .ff_filter("scale", 299, 299)  # or you can use '.scale(299, 299)'
            .start()  # this will return a packet generator
        )

        for i, pkt in enumerate(pkts):
            # convert frame to a nd array
            if pkt.is_(bmf.VideoFrame):
                vf = pkt.get(bmf.VideoFrame)
                rgb = mp.PixelInfo(mp.kPF_RGB24)
                np_vf = vf.reformat(rgb).frame().plane(0).numpy()
                # we can add some more processing here, e.g. predicting
                print("frame", i, "shape", np_vf.shape)
            else:
                break

    def test_generator_10_frame(self):
        pkts = (
            bmf.graph()
            .decode({"input_path": "../files/img.mp4"})["video"]
            .ff_filter("scale", 299, 299)  # or you can use '.scale(299, 299)'
            .start()  # this will return a packet generator
        )

        for i, pkt in enumerate(pkts):
            # convert frame to a nd array
            if pkt.is_(bmf.VideoFrame) and i < 10:
                vf = pkt.get(bmf.VideoFrame)
                rgb = mp.PixelInfo(mp.kPF_RGB24)
                np_vf = vf.reformat(rgb).frame().plane(0).numpy()
                # we can add some more processing here, e.g. predicting
                print("frame", i, "shape", np_vf.shape)
            else:
                break


if __name__ == "__main__":
    unittest.main()
