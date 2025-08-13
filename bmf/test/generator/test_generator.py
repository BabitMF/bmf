import sys
import time
import unittest

sys.path.append("../../..")
import bmf
import bmf.hmp as mp
import os
if os.name == 'nt':
    # We redefine timeout_decorator on windows
    class timeout_decorator:

        @staticmethod
        def timeout(*args, **kwargs):
            return lambda f: f  # return a no-op decorator
else:
    import timeout_decorator

sys.path.append("../../test/")
from base_test.base_test_case import BaseTestCase
from base_test.media_info import MediaInfo


class TestGenerator(BaseTestCase):

    @timeout_decorator.timeout(seconds=120)
    def test_generator(self):
        pkts = (
            bmf.graph().decode({
                'input_path':
                "../../files/big_bunny_10s_30fps.mp4"
            })['video'].ff_filter('scale', 299,
                                  299)  # or you can use '.scale(299, 299)'
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
            bmf.graph().decode({
                'input_path':
                "../../files/big_bunny_10s_30fps.mp4"
            })['video'].ff_filter('scale', 299,
                                  299)  # or you can use '.scale(299, 299)'
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

    def test_multistream_async_generator(self):
        graph = bmf.graph({
            "dump_graph": 1
        })
        output_streams = graph.decode({
                'input_path':
                "../../files/big_bunny_10s_30fps.mp4"
            })
        v = output_streams['video']
        a = output_streams['audio']
        graph.start_multiple_streams([v, a])
        video_pkt_list = []
        audio_pkt_list = []
        v_eof = False
        a_eof = False
        while not v_eof or not a_eof:
            pkt_v = graph.poll_packet(v.get_name(), False)
            pkt_a = graph.poll_packet(a.get_name(), False)
            if pkt_v.defined() and pkt_v.timestamp == bmf.Timestamp.EOF:
                v_eof = True
                print("video eof")
            if pkt_a.defined() and pkt_a.timestamp == bmf.Timestamp.EOF:
                a_eof = True
                print("audio eof")
            has_audio_data = False
            has_video_data = False
            if pkt_v.is_(bmf.VideoFrame):
                video_pkt_list.append(pkt_v)
                has_video_data = True
            
            if pkt_a.is_(bmf.AudioFrame):
                audio_pkt_list.append(pkt_a)
                has_audio_data = True

            # 情况1: 未获取到任何数据，继续轮询
            if not has_video_data and not has_audio_data:
                continue
            if has_video_data:
                print("has_video_data")
                while True:
                    pkt = graph.poll_packet(v.get_name(), False)
                    if pkt.is_(bmf.VideoFrame):
                        print("video frame")
                        video_pkt_list.append(pkt)
                    elif pkt.defined() and pkt.timestamp == bmf.Timestamp.EOF:
                        v_eof = True
                        print("video eof")
                        break
                    else:
                        break
                
                print("video packet over len of video_pkt_list", len(video_pkt_list))
                video_pkt_list = []
            if has_audio_data:
                print("has_audio_data")
                while True:
                    pkt = graph.poll_packet(a.get_name(), False)
                    if pkt.is_(bmf.AudioFrame):
                        print("audio frame")
                        audio_pkt_list.append(pkt)
                    elif pkt.defined() and pkt.timestamp == bmf.Timestamp.EOF:
                        a_eof = True
                        print("audio eof")
                        break
                    else:
                        print("audio packet over len of audio_pkt_list", len(audio_pkt_list))
                        audio_pkt_list = []
                        break

if __name__ == "__main__":
    unittest.main()
