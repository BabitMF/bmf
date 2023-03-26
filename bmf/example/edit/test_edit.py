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


class TestEditModule(BaseTestCase):
    @timeout_decorator.timeout(seconds=120)
    def test_video_overlays(self):
        input_video_path = "../files/img.mp4"
        logo_path = "../files/xigua_prefix_logo_x.mov"
        output_path = "./overlays.mp4"
        expect_result = '../edit/overlays.mp4|480|640|6.984000|MOV,MP4,M4A,3GP,3G2,MJ2|989234|863602|h264|' \
                        '{"fps": "30.0715990453"}'
        self.remove_result_data(output_path)
        dump_graph = 0
        # create graph
        my_graph = bmf.graph({
            "dump_graph": dump_graph
        })

        overlay_option = {
            "dump_graph": dump_graph,
            "source": {
                "start": 0,
                "duration": 7,
                "width": 1280,
                "height": 720
            },
            "overlays": [
                {
                    "start": 0,
                    "duration": 4,
                    "width": 300,
                    "height": 200,
                    "pox_x": 0,
                    "pox_y": 0,
                    "loop": 0,
                    "repeat_last": 1
                }
            ]
        }

        # main video
        video = my_graph.decode({'input_path': input_video_path})

        # logo video
        logo_1 = my_graph.decode({'input_path': logo_path})['video']

        # call 'my_overlay' module to do overlay
        output_streams = (
            bmf.module([video['video'], logo_1], 'video_overlay', overlay_option)
        )

        (
            output_streams[0].encode(None, {
                "output_path": output_path,
                "video_params": {
                    "width": 640,
                    "height": 480,
                    'codec': 'h264'
                }
            }).run()
        )
        self.check_video_diff(output_path, expect_result)

    @timeout_decorator.timeout(seconds=120)
    def test_audio_mix(self):
        input_video_path = "../files/img.mp4"
        output_path = "./audio_mix.mp4"
        expect_result = '../edit/audio_mix.mp4|480|640|7.550000|MOV,MP4,M4A,3GP,3G2,MJ2|1143753|1079417|h264|' \
                        '{"fps": "30.0662251656"}'
        self.remove_result_data(output_path)
        dump_graph = 0
        # create graph
        my_graph = bmf.graph()

        option = {
            "audios": [
                {
                    "start": 1,
                    "duration": 5
                },
                {
                    "start": 0,
                    "duration": 5
                }
            ]
        }

        # main video
        video = my_graph.decode({'input_path': input_video_path})
        video2 = my_graph.decode({'input_path': input_video_path})

        # mix audio
        audio_stream = (
            bmf.module([video['audio'], video2['audio']], 'audio_mix', option)
        )

        # encode
        (
            video['video'].encode(audio_stream, {
                "output_path": output_path,
                "video_params": {
                    "width": 640,
                    "height": 480
                }
            }).run()
        )
        self.check_video_diff(output_path, expect_result)

    @timeout_decorator.timeout(seconds=120)
    def test_video_concat(self):
        input_video_path = "../files/img.mp4"
        output_path = "./video_concat.mp4"
        expect_result = '../edit/video_concat.mp4|720|1280|15.022000|MOV,MP4,M4A,3GP,3G2,MJ2|2959335|5556892|h264|' \
                        '{"fps": "30.0166759311"}'
        self.remove_result_data(output_path)
        dump_graph = 0
        # create graph
        my_graph = bmf.graph({
            "dump_graph": dump_graph
        })

        option = {
            "dump_graph": dump_graph,
            "width": 1280,
            "height": 720,
            # if have audio input
            "has_audio": 1,
            "video_list": [
                {
                    "start": 0,
                    "duration": 7,
                    "transition_time": 2,
                    "transition_mode": 1
                },
                {
                    "start": 0,
                    "duration": 7,
                    "transition_time": 2,
                    "transition_mode": 1
                },
                {
                    "start": 0,
                    "duration": 7,
                    "transition_time": 2,
                    "transition_mode": 1
                }
            ]
        }

        # main video
        video1 = my_graph.decode({'input_path': input_video_path})

        video2 = my_graph.decode({'input_path': input_video_path})

        video3 = my_graph.decode({'input_path': input_video_path})

        # concat video and audio streams with 'video_concat' module
        concat_streams = (
            bmf.module([
                video1['video'],
                video2['video'],
                video3['video'],
                video1['audio'],
                video2['audio'],
                video3['audio']
            ], 'video_concat', option)
        )

        # encode
        (
            bmf.encode(concat_streams[0], concat_streams[1], {
                "output_path": output_path,
                "video_params": {
                    "width": 1280,
                    "height": 720
                }
            }).run()
        )

        self.check_video_diff(output_path, expect_result)

    @timeout_decorator.timeout(seconds=120)
    def test_complex_edit(self):
        input_video_path = "../files/img.mp4"
        logo_path = "../files/xigua_prefix_logo_x.mov"
        output_path = "./complex_edit.mp4"
        expect_result = '../edit/complex_edit.mp4|720|1280|15.022000|MOV,MP4,M4A,3GP,3G2,MJ2|2987997|5610712|h264|' \
                        '{"fps": "30.0166759311"}'
        self.remove_result_data(output_path)
        dump_graph = 0
        # create graph
        duration = 7

        overlay_option = {
            "dump_graph": dump_graph,
            "source": {
                "start": 0,
                "duration": duration,
                "width": 1280,
                "height": 720
            },
            "overlays": [
                {
                    "start": 0,
                    "duration": duration,
                    "width": 300,
                    "height": 200,
                    "pox_x": 0,
                    "pox_y": 0,
                    "loop": 0,
                    "repeat_last": 1
                }
            ]
        }

        concat_option = {
            "dump_graph": dump_graph,
            "width": 1280,
            "height": 720,
            # if have audio input
            "has_audio": 1,
            "video_list": [
                {
                    "start": 0,
                    "duration": duration,
                    "transition_time": 2,
                    "transition_mode": 1
                },
                {
                    "start": 0,
                    "duration": duration,
                    "transition_time": 2,
                    "transition_mode": 1
                },
                {
                    "start": 0,
                    "duration": duration,
                    "transition_time": 2,
                    "transition_mode": 1
                }
            ]
        }

        # create graph
        my_graph = bmf.graph({
            "dump_graph": dump_graph
        })

        # three logo video
        logo_1 = my_graph.decode({'input_path': logo_path})['video']
        logo_2 = my_graph.decode({'input_path': logo_path})['video']
        logo_3 = my_graph.decode({'input_path': logo_path})['video']

        # three videos
        video1 = my_graph.decode({'input_path': input_video_path})
        video2 = my_graph.decode({'input_path': input_video_path})
        video3 = my_graph.decode({'input_path': input_video_path})

        # do overlay
        overlay_streams = list()
        overlay_streams.append(bmf.module([video1['video'], logo_1], 'video_overlay', overlay_option)[0])
        overlay_streams.append(bmf.module([video2['video'], logo_2], 'video_overlay', overlay_option)[0])
        overlay_streams.append(bmf.module([video3['video'], logo_3], 'video_overlay', overlay_option)[0])

        # do concat
        concat_streams = (
            bmf.module([
                overlay_streams[0],
                overlay_streams[1],
                overlay_streams[2],
                video1['audio'],
                video2['audio'],
                video3['audio']
            ], 'video_concat', concat_option)
        )

        # encode
        (
            bmf.encode(concat_streams[0], concat_streams[1], {
                "output_path": output_path,
                "video_params": {
                    "width": 1280,
                    "height": 720,
                    'codec': 'h264'
                }
            }).run()
        )
        self.check_video_diff(output_path, expect_result)


if __name__ == '__main__':
    unittest.main()
