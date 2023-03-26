import sys
import time
import unittest

sys.path.append("../../")
import bmf
import timeout_decorator

sys.path.append("../")
from base_test.base_test_case import BaseTestCase


class TestTranscode(BaseTestCase):
    def test_audio(self):
        input_video_path = "../files/img.mp4"
        output_path = "./audio.mp4"
        # create graph
        graph = bmf.graph()

        # decode
        video = graph.decode({
            "input_path": input_video_path
        })

        (
            bmf.encode(
                None,
                video['audio'],
                {
                    "output_path": output_path,
                    "audio_params": {
                        "codec": "aac",
                        "bit_rate": 128000,
                        "sample_rate": 44100,
                        "channels": 2
                    }
                }
            )
                .run()
        )

    def test_with_input_only_audio(self):
        input_video_path = "../files/img_a.mp4"
        output_path = "./output.mp4"
        mygraph = bmf.graph()
        streams = mygraph.decode({'input_path': input_video_path})
        bmf.encode(None,streams["audio"], {"output_path": output_path}).run()

    def test_with_encode_with_audio_stream_but_no_audio_frame(self):
        input_video_path = "../files/img_v.mp4"
        output_path = "./output.mp4"
        mygraph = bmf.graph()
        streams = mygraph.decode({'input_path': input_video_path})
        bmf.encode(streams["video"],streams["audio"], {"output_path": output_path}).run()

    def test_with_null_audio(self):
        input_video_path = "../files/img.mp4"
        output_path = "./with_null_audio.mp4"
        # create graph
        graph = bmf.graph()

        # decode
        video = graph.decode({
            "input_path": input_video_path
        })

        # create a null audio stream
        audio_stream = graph.anullsrc('r=48000', 'cl=2').atrim('start=0', 'end=0.3')

        (
            bmf.encode(
                video['video'],
                audio_stream,
                {
                    "output_path": output_path,
                    "video_params": {
                        "codec": "h264",
                        "width": 320,
                        "height": 240,
                        "crf": 23,
                        "preset": "veryfast"
                    },
                    "audio_params": {
                        "codec": "aac",
                        "bit_rate": 128000,
                        "sample_rate": 44100,
                        "channels": 2
                    }
                }
            )
                .run()
        )

    def test_simple(self):
        input_video_path = "../files/img.mp4"
        output_path = "./simple.mp4"

        # create graph
        graph = bmf.graph()

        # decode
        video = graph.decode({
            "input_path": input_video_path
        })

        (
            bmf.encode(
                video['video'],
                video['audio'],
                {
                    "output_path": output_path,
                    "video_params": {
                        "codec": "h264",
                        "width": 320,
                        "height": 240,
                        "crf": 23,
                        "preset": "veryfast"
                    },
                    "audio_params": {
                        "codec": "aac",
                        "bit_rate": 128000,
                        "sample_rate": 44100,
                        "channels": 2
                    }
                }
            )
                .run()
        )

    def test_cb(self):
        input_video_path = "../files/img.mp4"
        output_path = "./cb.mp4"

        # create graph
        graph = bmf.graph()

        def cb(para):
            print(para)
            return bytes("OK","ASCII")

        graph.add_user_callback(bmf.BmfCallBackType.LATEST_TIMESTAMP, cb)

        # decode
        video = graph.decode({
            "input_path": input_video_path
        })

        bmf.encode(
            video['video'],
            video['audio'],
            {
                "output_path": output_path,
                "video_params": {
                    "codec": "h264",
                    "width": 320,
                    "height": 240,
                    "crf": "23",
                    "preset": "veryfast"
                }
            }
        ).node_.add_user_callback(bmf.BmfCallBackType.LATEST_TIMESTAMP,cb)
        graph.run()

    def test_option(self):
        input_video_path = "../files/img.mp4"
        output_path = "./option.mp4"

        # create graph
        graph = bmf.graph()

        # decode
        # start_time in seconds
        video = graph.decode({
            "input_path": input_video_path,
        })

        (
            bmf.encode(
                video['video'],
                video['audio'],
                {
                    "output_path": output_path,
                    "video_params": {
                        "codec": "h264",
                        "width": 1280,
                        "height": 720,
                        "preset": "fast",
                        "crf": "23",
                        "x264-params": "ssim=1:psnr=1"
                    },
                    "audio_params": {
                        "codec": "aac",
                        "bit_rate": 128000,
                        "sample_rate": 44100,
                        "channels": 2
                    },
                    "mux_params": {
                        "fflags": "+igndts",
                        "movflags": "+faststart+use_metadata_tags",
                        "max_interleave_delta": "0"
                    }
                }
            ).run()
        )

    def test_image(self):
        input_video_path = "../files/overlay.png"
        output_path = "./image.jpg"
        (
            bmf.graph()
                .decode({'input_path': input_video_path})['video']
                .scale(320, 240)
                .encode(None, {
                "output_path": output_path,
                "format": "mjpeg",
                "video_params": {
                    "codec": "jpg",
                    "width": 320,
                    "height": 240
                }
            }).run()
        )

    def test_concat_video_and_audio(self):
        input_video_path = '../files/img.mp4'
        output_path = "./concat_video_and_audio.mp4"

        # create graph
        my_graph = bmf.graph()

        # decode video
        video1 = my_graph.decode({'input_path': input_video_path})
        video2 = my_graph.decode({'input_path': input_video_path})

        # do concat
        concat_video_stream = (
            bmf.concat(
                video1['video'],
                video2['video'],
            )
        )

        concat_audio_stream = (
            bmf.concat(
                video1['audio'],
                video2['audio'],
                v=0,
                a=1
            )
        )

        # encode
        (
            bmf.encode(
                concat_video_stream,
                concat_audio_stream,
                {
                    "output_path": output_path
                }
            ).run()
        )

    def test_short_video_concat(self):
        input_video_path = "../files/img.mp4"
        input_video_path2 = "../files/single_frame.mp4"

        output_path = "./short_video_concat.mp4"
        # create graph
        graph = bmf.graph()

        # create graph
        graph = bmf.graph({'dump_graph': 1})

        # decode
        video = graph.decode({
            "input_path": input_video_path
        })['video']
        video2 = graph.decode({
            "input_path": input_video_path2
        })['video']

        vout = video.concat(video2)

        (
            bmf.encode(
                vout,
                None,
                {
                    "output_path": output_path,
                    "video_params": {
                        "codec": "h264",
                        "width": 320,
                        "height": 240,
                        "crf": 23,
                        "preset": "veryfast"
                    }
                }
            )
                .run()
        )


if __name__ == '__main__':
    unittest.main()
