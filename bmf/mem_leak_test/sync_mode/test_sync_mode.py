import sys
import unittest

sys.path.append("../../..")
import bmf
from bmf import bmf_sync, Packet
import timeout_decorator

sys.path.append("../")
from base_test.base_test_case import BaseTestCase


class TestSyncMode(BaseTestCase):
    def test_videoframe(self):
        input_video_path = "../files/overlay.png"
        output_path = "./videoframe.jpg"

        # create decoder
        decoder = bmf_sync.sync_module("c_ffmpeg_decoder", {"input_path": input_video_path}, [], [0])

        # create scale
        scale = bmf_sync.sync_module("c_ffmpeg_filter", {
            "name": "scale",
            "para": "320:240"
        }, [0], [0])

        # create encoder
        encoder = bmf_sync.sync_module("c_ffmpeg_encoder", {
            "output_path": output_path,
            "format": "mjpeg",
            "video_params": {
                "codec": "jpg"
            }
        }, [0], [])

        # decode
        frames, _ = bmf_sync.process(decoder, None)

        # scale
        frames, _ = bmf_sync.process(scale, {0:frames[0]})

        # encode
        bmf_sync.process(encoder, {0:frames[0]})

        # send eof to encoder
        bmf_sync.send_eof(encoder)

    def test_videoframe_by_graph(self):
        input_video_path = "../files/overlay.png"
        output_path = "./videoframe_by_graph.jpg"

        graph = bmf.graph()
        video = graph.decode({
            "alias": "decoder",
            "input_path": input_video_path
        })['video']

        video = video.scale(320, 240, alias="scale")

        bmf.encode(video, None, {
            "alias": "encoder",
            "output_path": output_path,
            "format": "mjpeg",
            "video_params": {
                "codec": "jpg"
            }
        })

        # create sync modules
        decoder = graph.get_module("decoder")
        scale = graph.get_module("scale")
        encoder = graph.get_module("encoder")

        # decode and get video frame
        frames, _ = bmf_sync.process(decoder, None)

        # scale
        frames, _ = bmf_sync.process(scale, {0: frames[0]})

        # encode
        bmf_sync.process(encoder, {0: frames[0]})

        # send eof to encoder
        bmf_sync.send_eof(encoder)

    def test_audioframe(self):
        input_video_path = "../files/img.mp4"
        output_path = "./audioframe.mp4"

        # create decoder
        decoder = bmf_sync.sync_module("c_ffmpeg_decoder", {"input_path": input_video_path}, [], [1])

        # create volume
        volume = bmf_sync.sync_module("c_ffmpeg_filter", {
            "name": "volume",
            "para": "volume=3"
        }, [0], [0])

        # create encoder
        encoder = bmf_sync.sync_module("c_ffmpeg_encoder", {
            "output_path": output_path
        }, [0, 1], [])

        # ecode and get audio frame
        frames, _ = bmf_sync.process(decoder, None)

        # volume filter
        frames, _ = bmf_sync.process(volume, {0:frames[1]})

        # encode
        bmf_sync.process(encoder, {1:frames[0]})

        # send eof to encoder
        bmf_sync.send_eof(encoder)

    def test_audioframe_by_graph(self):
        input_video_path = "../files/img.mp4"
        output_path = "./audioframe_by_graph.mp4"

        graph = bmf.graph()
        audio = graph.decode({
            "alias": "decoder",
            "input_path": input_video_path
        })['audio']

        audio = audio.ff_filter('volume', volume=3, alias="volume")

        bmf.encode(None, audio, {
            "alias": "encoder",
            "output_path": output_path,
        })

        # create sync modules
        decoder = graph.get_module("decoder")
        volume = graph.get_module("volume")
        encoder = graph.get_module("encoder")

        # ecode and get audio frame
        frames, _ = bmf_sync.process(decoder, None)

        # volume filter
        frames, _ = bmf_sync.process(volume, {0: frames[1]})

        # encode
        bmf_sync.process(encoder, {1: frames[0]})

        # send eof to encoder
        bmf_sync.send_eof(encoder)

    def test_video(self):
        input_video_path = "../files/img.mp4"
        output_path = "./video.mp4"

        # create sync modules
        decoder = bmf_sync.sync_module("c_ffmpeg_decoder", {"input_path": input_video_path}, [], [0, 1])
        scale = bmf_sync.sync_module("c_ffmpeg_filter", {
            "name": "scale",
            "para": "320, 250"
        }, [0], [0])
        volume = bmf_sync.sync_module("c_ffmpeg_filter", {
            "name": "volume",
            "para": "volume=3"
        }, [0], [0])
        encoder = bmf_sync.sync_module("c_ffmpeg_encoder", {
            "output_path": output_path
        }, [0, 1], [])

        # process video/audio by sync mode
        while True:
            frames, _ = bmf_sync.process(decoder, None)
            has_next = False
            for key in frames:
                if len(frames[key]) > 0:
                    has_next = True
                    break
            if not has_next:
                bmf_sync.send_eof(encoder)
                break
            if 0 in frames.keys() and len(frames[0]) > 0:
                frames, _ = bmf_sync.process(scale, {0: frames[0]})
                bmf_sync.process(encoder, {0: frames[0]})
            if 1 in frames.keys() and len(frames[1]) > 0:
                frames, _ = bmf_sync.process(volume, {0: frames[1]})
                bmf_sync.process(encoder, {1: frames[0]})

    def test_video_by_graph(self):
        input_video_path = "../files/img.mp4"
        output_path = "./video_by_graph.mp4"

        graph = bmf.graph()

        video = graph.decode({
            "alias": "decoder",
            "input_path": input_video_path
        })

        v = video['video'].scale(320, 250, alias="scale")
        a = video['audio'].ff_filter('volume', volume=3, alias="volume")

        bmf.encode(v, a, {
            "alias": "encoder",
            "output_path": output_path
        })

        # create sync modules
        decoder = graph.get_module("decoder")
        scale = graph.get_module("scale")
        volume = graph.get_module("volume")
        encoder = graph.get_module("encoder")

        # process video/audio by sync mode
        while True:
            frames, _ = bmf_sync.process(decoder, None)
            has_next = False
            for key in frames:
                if len(frames[key]) > 0:
                    has_next = True
                    break
            if not has_next:
                bmf_sync.send_eof(encoder)
                break
            if 0 in frames.keys() and len(frames[0]) > 0:
                frames, _ = bmf_sync.process(scale, {0: frames[0]})
                bmf_sync.process(encoder, {0: frames[0]})
            if 1 in frames.keys() and len(frames[1]) > 0:
                frames, _ = bmf_sync.process(volume, {0: frames[1]})
                bmf_sync.process(encoder, {1: frames[0]})

    def test_video_only_situation(self):
        input_video_path = "../files/img.mp4"
        output_path = "./video_only_situation.mp4"

        graph = bmf.graph()

        video = graph.decode({
            "alias": "decoder",
            "input_path": input_video_path
        })

        v = video['video'].scale(320, 250, alias="scale")

        bmf.encode(v, None, {
            "alias": "encoder",
            "output_path": output_path
        })

        # create sync modules
        decoder = graph.get_module("decoder")
        scale = graph.get_module("scale")
        encoder = graph.get_module("encoder")

        # process video by sync mode
        while True:
            frames, _ = bmf_sync.process(decoder, None)
            has_next = False
            for key in frames:
                if len(frames[key]) > 0:
                    has_next = True
                    break
            if not has_next:
                bmf_sync.send_eof(encoder)
                break
            if 0 in frames.keys() and len(frames[0]) > 0:
                frames, _ = bmf_sync.process(scale, {0: frames[0]})
                bmf_sync.process(encoder, {0: frames[0]})

    def test_audio(self):
        input_video_path = "../files/img.mp4"
        output_path = "./audio.mp4"

        # create sync modules
        decoder = bmf_sync.sync_module("c_ffmpeg_decoder", {"input_path": input_video_path}, [], [1])
        volume = bmf_sync.sync_module("c_ffmpeg_filter", {
            "name": "volume",
            "para": "volume=3"
        }, [0], [0])
        encoder = bmf_sync.sync_module("c_ffmpeg_encoder", {
            "output_path": output_path
        }, [0, 1], [])

        # process audio by sync mode
        while True:
            frames, _ = bmf_sync.process(decoder, None)
            if len(frames[1]) == 0:
                bmf_sync.send_eof(encoder)
                break
            else:
                frames, _ = bmf_sync.process(volume, {0: frames[1]})
                bmf_sync.process(encoder, {1: frames[0]})

    def test_audio_by_graph(self):
        input_video_path = "../files/img.mp4"
        output_path = "./audio_by_graph.mp4"

        graph = bmf.graph()

        audio = graph.decode({
            "alias": "decoder",
            "input_path": input_video_path
        })['audio'].ff_filter('volume', volume=3, alias="volume")

        bmf.encode(None, audio, {
            "alias": "encoder",
            "output_path": output_path
        })

        # create sync modules
        decoder = graph.get_module("decoder")
        volume = graph.get_module("volume")
        encoder = graph.get_module("encoder")

        # process audio by sync mode
        while True:
            frames, _ = bmf_sync.process(decoder, None)
            if len(frames[1]) == 0:
                bmf_sync.send_eof(encoder)
                break
            else:
                frames, _ = bmf_sync.process(volume, {0: frames[1]})
                bmf_sync.process(encoder, {1: frames[0]})


if __name__ == '__main__':
    unittest.main()
