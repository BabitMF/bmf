import sys
import time
import unittest

sys.path.append("../../")
import bmf
from bmf import Log, LogLevel
import timeout_decorator

sys.path.append("../../bmf/example/")
from base_test.base_test_case import BaseTestCase
from base_test.media_info import MediaInfo


class TestComplexCase(BaseTestCase):
    @timeout_decorator.timeout(seconds=120)
    def test_concat_n_videos(self):

        input_video_path = "../files/big_bunny_10s_30fps.mp4"
        output_path = "./concat_n_videos.mp4"
        expect_result = '../../../test/case/concat_n_videos.mp4|720|1280|50.036|MOV,MP4,M4A,3GP,3G2,MJ2|' \
                        '837866|5240430|h264|{"fps": "29.9867899604"}'
        self.remove_result_data(output_path)

        n = 5
        my_graph = bmf.graph()
        video_list = []
        for i in range(n):
            video1 = my_graph.decode({'input_path': input_video_path})
            video_list.append(video1)

        concat_video_streams = []
        concat_audio_streams = []

        times = n

        # concat several times
        for i in range(times):
            # concat video
            concat_video_streams.append(video_list[i]['video'])
            # concat audio
            concat_audio_streams.append(video_list[i]['audio'])

        concat_video_stream = bmf.concat(*concat_video_streams, n=times, v=1, a=0)
        concat_audio_stream = bmf.concat(*concat_audio_streams, n=times, v=0, a=1)

        # encode
        bmf.encode(concat_video_stream,
                   concat_audio_stream,
                   {
                       "output_path": output_path,
                       "video_params": {
                           "width": 1280,
                           "height": 720
                       }
                   }).run()

        self.check_video_diff(output_path, expect_result)

    # @timeout_decorator.timeout(seconds=120)
    # def test_concat_only_video(self):
    #
    #     input_video_path_1 = "../files/normal.mp4"
    #     input_video_path_2 = "../files/only_video.mp4"
    #     output_path = "./concat_only_video.mp4"
    #     expect_result = '../../../test/case/edit_concat.mp4|720|1280|29.959000|MOV,MP4,M4A,3GP,3G2,MJ2|987564|' \
    #                     '3698307|h264|{"fps": "20.0278164117"}'
    #     self.remove_result_data(output_path)
    #
    #     my_graph = bmf.graph({"dump_graph": 1})
    #     video1 = my_graph.decode({'input_path': input_video_path_1})
    #     video2 = my_graph.decode({'input_path': input_video_path_2})
    #
    #     concat_video_streams = []
    #     concat_audio_streams = []
    #
    #     # concat video
    #     concat_video_streams.append(video1['video'].scale(1280, 720))
    #     concat_video_streams.append(video2['video'].scale(1280, 720))
    #     # concat audio
    #     concat_audio_streams.append(video1['audio'])
    #     concat_audio_streams.append(video2['audio'])
    #
    #     concat_video_stream = bmf.concat(*concat_video_streams, v=1, a=0)
    #     concat_audio_stream = bmf.concat(*concat_audio_streams, v=0, a=1)
    #
    #     # encode
    #     bmf.encode(concat_video_stream,
    #                concat_audio_stream,
    #                {
    #                    "output_path": output_path,
    #                    "video_params": {
    #                        "width": 1280,
    #                        "height": 720
    #                    }
    #                }).run()
    #
    #     self.check_video_diff(output_path, expect_result)

    # @timeout_decorator.timeout(seconds=120)
    # def test_concat_only_audio(self):
    #
    #     input_video_path_1 = "../files/normal.mp4"
    #     input_video_path_2 = "../files/only_audio.mp4"
    #     output_path = "./test_concat_only_audio.mp4"
    #     expect_result = '../../../test/case/edit_concat.mp4|720|1280|29.959000|MOV,MP4,M4A,3GP,3G2,MJ2|987564|' \
    #                     '3698307|h264|{"fps": "20.0278164117"}'
    #     self.remove_result_data(output_path)
    #
    #     my_graph = bmf.graph()
    #     video1 = my_graph.decode({'input_path': input_video_path_1})
    #     video2 = my_graph.decode({'input_path': input_video_path_2})
    #
    #     concat_video_streams = []
    #     concat_audio_streams = []
    #
    #     # concat video
    #     concat_video_streams.append(video1['video'].scale(1280, 720))
    #     concat_video_streams.append(video2['video'].scale(1280, 720))
    #     # concat audio
    #     concat_audio_streams.append(video1['audio'])
    #     concat_audio_streams.append(video2['audio'])
    #
    #     concat_video_stream = bmf.concat(*concat_video_streams, v=1, a=0)
    #     concat_audio_stream = bmf.concat(*concat_audio_streams, v=0, a=1)
    #
    #     # encode
    #     bmf.encode(concat_video_stream,
    #                concat_audio_stream,
    #                {
    #                    "output_path": output_path,
    #                    "video_params": {
    #                        "width": 1280,
    #                        "height": 720
    #                    }
    #                }).run()
    #
    #     self.check_video_diff(output_path, expect_result)

    @timeout_decorator.timeout(seconds=120)
    def test_trim_and_concat_n_videos(self):

        input_video_path = "../files/big_bunny_10s_30fps.mp4"
        output_path = "./trim_and_concat_n_videos.mp4"
        expect_result = '../../../test/case/trim_and_concat_n_videos.mp4|720|1280|10.022000|MOV,MP4,M4A,3GP,3G2,MJ2|' \
                        '318221|397777|h264|{"fps": "30.0500834725"}'
        self.remove_result_data(output_path)

        n = 5
        # create graph
        my_graph = bmf.graph()

        option = {
            "width": 1280,
            "height": 720,
            # if have audio input
            "has_audio": 1,
            "video_list": []
        }

        for i in range(n):
            video_info = {
                "start": 0,
                "duration": 3,
                "transition_time": 1,
                "transition_mode": 1
            }
            option.get("video_list").append(video_info)

        video_list = []
        for i in range(n):
            video1 = my_graph.decode({'input_path': input_video_path})
            video_list.append(video1)

        stream = []
        for i in range(n):
            stream.append(video_list[i]['video'])
        for i in range(n):
            stream.append(video_list[i]['audio'])

        # concat video and audio streams with 'video_concat' module
        concat_streams = (
            bmf.module(stream, 'video_concat', option)
        )

        # encode
        (
            bmf.encode(concat_streams[0], concat_streams[1],
                       {
                           "output_path": output_path,
                           "video_params": {
                               "width": 1280,
                               "height": 720
                           }
                       })
                .run()
        )
        self.check_video_diff(output_path, expect_result)

    @timeout_decorator.timeout(seconds=120)
    def test_subgraph_serial(self):

        input_video_path = "../files/big_bunny_10s_30fps.mp4"
        logo_path = "../files/xigua_prefix_logo_x.mov"
        output_path = "./subgraph_serial.mp4"
        expect_result = '../../../test/case/subgraph_serial.mp4|720|1280|6.022000|MOV,MP4,M4A,3GP,3G2,MJ2|' \
                        '358204|268653|h264|{"fps": "30.0835654596"}'
        self.remove_result_data(output_path)

        # if dump graph to json file
        dump_graph = 0

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
                    "duration": 7,
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
                    "duration": 3,
                    "transition_time": 1,
                    "transition_mode": 1
                },
                {
                    "start": 0,
                    "duration": 3,
                    "transition_time": 1,
                    "transition_mode": 1
                },
                {
                    "start": 0,
                    "duration": 3,
                    "transition_time": 1,
                    "transition_mode": 1
                }
            ]
        }

        # create graph
        my_graph = bmf.graph({
            "dump_graph": dump_graph
        })

        # three videos
        video1 = my_graph.decode({'input_path': input_video_path})
        video2 = my_graph.decode({'input_path': input_video_path})
        video3 = my_graph.decode({'input_path': input_video_path})

        # three logo video
        logo_1 = my_graph.decode({'input_path': logo_path})['video']
        logo_2 = my_graph.decode({'input_path': logo_path})['video']
        logo_3 = my_graph.decode({'input_path': logo_path})['video']

        logo_1 = logo_1.vflip()
        logo_2 = logo_2.vflip()

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
            bmf.encode(concat_streams[0], concat_streams[1],
                       {
                           "output_path": output_path,
                           "video_params": {
                               "width": 1280,
                               "height": 720
                           }
                       })
                .run()
        )
        self.check_video_diff(output_path, expect_result)

    @timeout_decorator.timeout(seconds=120)
    def test_subgraph_nested(self):

        input_video_path = "../files/big_bunny_10s_30fps.mp4"
        output_path = "./subgraph_nested.mp4"
        expect_result = '../../../test/case/subgraph_nested.mp4|720|1280|10.021000|MOV,MP4,M4A,3GP,3G2,MJ2|845735|' \
                        '1057169|h264|{"fps": "30.0"}'
        self.remove_result_data(output_path)

        my_graph = bmf.graph({"dump_graph": 1})
        video1 = my_graph.decode({'input_path': input_video_path})
        v1 = video1['video']
        a1 = video1['audio']
        concat_streams = (
            bmf.module([v1, a1], 'flip_and_scale', {"dump_graph": 1})
        )
        # encode
        (
            bmf.encode(concat_streams[0], concat_streams[1],
                       {
                           "output_path": output_path,
                           "video_params": {
                               "width": 1280,
                               "height": 720
                           }
                       }).run()
        )
        self.check_video_diff(output_path, expect_result)

    # @timeout_decorator.timeout(seconds=120)
    # def test_undealed_output(self):
    #
    #     input_video_path_1 = "../files/big_bunny_10s_30fps.mp4"
    #     input_video_path_2 = "../files/3min.mp4"
    #     output_path = "./undealed_output.mp4"
    #     expect_result = '../../../test/case/edit_concat.mp4|720|1280|29.959000|MOV,MP4,M4A,3GP,3G2,MJ2|987564|' \
    #                     '3698307|h264|{"fps": "20.0278164117"}'
    #     self.remove_result_data(output_path)
    #
    #     my_graph = bmf.graph()
    #     video1 = my_graph.decode({'input_path': input_video_path_1})
    #     video2 = my_graph.decode({'input_path': input_video_path_2})
    #     v1 = video1['video'].scale(300, 300)
    #     a1 = video1['audio']
    #     v2 = video2['video'].scale(300, 300).trim(start=0, end=5)
    #     a2 = video2['audio'].atrim(start=0, end=5)
    #     concat_video_stream = bmf.concat(v1, v2, n=2, v=1, a=0)
    #     concat_audio_stream = bmf.concat(a1, a2, n=2, v=0, a=1)
    #     bmf.encode(concat_video_stream,
    #                None,
    #                {
    #                    "output_path": output_path,
    #                    "video_params": {
    #                        "width": 1280,
    #                        "height": 720
    #                    }
    #                }).run()
    #     self.check_video_diff(output_path, expect_result)


if __name__ == '__main__':
    Log.set_log_level(LogLevel.DEBUG)
    unittest.main()
