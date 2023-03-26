import sys
import time
import unittest

sys.path.append("../../..")
import bmf
import timeout_decorator

sys.path.append("../")
from base_test.base_test_case import BaseTestCase
from base_test.media_info import MediaInfo


class TestVideoEditPrototype(BaseTestCase):
    @timeout_decorator.timeout(seconds=120)
    def test_video_preprocess(self):
        input_video_path = "../files/img.mp4"
        output_path = "./video_preprocess.mp4"
        expect_result = '../video_edit_prototype/video_preprocess.mp4|480|640|5.014000|MOV,MP4,M4A,3GP,3G2,MJ2|2114359|' \
                        '1325175|h264|{"fps": "30.1003344482"}'
        self.remove_result_data(output_path)
        # create graph
        graph = bmf.graph({'dump_graph': 1})

        # decode
        video = graph.decode({
            "input_path": input_video_path
        })

        option = {
            'Type': 'video',
            'Source': input_video_path,
            'StartTime': 2,
            'Duration': 5,
            'Position': {
                'PosX': 0,
                'PosY': 0,
                'Width': 1280,
                'Height': 720
            },
            'Crop': {
                'PosX': 0,
                'PosY': 0,
                'Width': 1280,
                'Height': 720
            },
            'Speed': 1.0,
            'Rotate': 90,
            'Delogo': {
                'PosX': '2%',
                'PosY': '2%',
                'Width': '400',
                'Height': '200'
            },
            'Trims': [
                [0, 1],
                [3, 7]
            ],
            "Filters": {
                "Contrast": 110,
                "Brightness": 62,
                "Saturate": 121,
                "Opacity": 66,
                "Blur": 28
            },
            'ExtraFilters': [
                {
                    'Type': 'AAA',
                    'Para': 'BBB',
                }
            ],
            'Volume': 0.8,
            'Mute': 0
        }

        init_info = {}
        init_info['video_duration'] = 5
        init_info['audio_duration'] = 5
        option['initial_info'] = init_info

        processed_video = bmf.module([video['video'], video['audio']], 'video_preprocess', option)

        (
            bmf.encode(
                processed_video[0],
                processed_video[1],
                {
                    "output_path": output_path,
                    "video_params": {
                        "codec": "h264",
                        "width": 640,
                        "height": 480,
                        "crf": "23",
                        "preset": "veryfast"
                    }
                }
            ).run()
        )
        self.check_video_diff(output_path, expect_result)

    @timeout_decorator.timeout(seconds=120)
    def test_audio_preprocess(self):
        input_video_path = "../files/img.mp4"
        output_path = "./audio_preprocess.mp4"
        expect_result = '../video_edit_prototype/audio_preprocess.mp4|480|640|7.550000|MOV,MP4,M4A,3GP,3G2,MJ2|996517|' \
                        '940463|h264|{"fps": "30.0662251656"}'
        self.remove_result_data(output_path)

        # create graph
        graph = bmf.graph({'dump_graph': 1})

        # decode
        video = graph.decode({
            "input_path": input_video_path
        })

        option = {
            'Type': 'audio',
            'Source': input_video_path,
            'StartTime': 0,
            'Duration': 5,
            'Trims': [
                [0, 1.2],
                [3, 5.5]
            ],
            'Volume': 0.1,
            'Loop': 0
        }
        init_info = {}
        init_info['audio_duration'] = 5
        option['initial_info'] = init_info

        audio = video['audio'].module('audio_preprocess', option)

        (
            bmf.encode(
                video['video'],
                audio,
                {
                    "output_path": output_path,
                    "video_params": {
                        "codec": "h264",
                        "width": 640,
                        "height": 480,
                        "crf": "23",
                        "preset": "veryfast"
                    }
                }
            ).run()
        )
        self.check_video_diff(output_path, expect_result)

    @timeout_decorator.timeout(seconds=120)
    def test_image_preprocess(self):
        input_video_path = "../files/blue.png"
        output_path = "./image_preprocess.png"
        expect_result = '../video_edit_prototype/image_preprocess.png|480|640|0.009000|MOV,MP4,M4A,3GP,3G2,MJ2|' \
                        '136269333|153303|png|{"fps": "120.0"}'
        self.remove_result_data(output_path)
        # create graph
        graph = bmf.graph({'dump_graph': 1})

        # decode
        video = graph.decode({
            "input_path": input_video_path
        })

        option = {
            'Type': 'image',
            'Source': input_video_path,
            'StartTime': 0,
            'Duration': 5,
            'Position': {
                'PosX': 512.0,
                'PosY': 0.0,
                'Width': 128.0,
                'Height': 96.0
            },
            'Crop': {
                'PosX': 25.6,
                'PosY': 19.2,
                'Width': 102.4,
                'Height': 76.8
            },
            'Rotate': 0,
            "Filters": {
                "Contrast": 110,
                "Brightness": 62,
                "Saturate": 121,
                "Opacity": 66,
                "Blur": 28
            },
            'ExtraFilters': {
                'Type': 'AAA',
                'Para': 'BBB',
            }
        }
        init_info = {}
        init_info['width'] = 1438.0
        init_info['height'] = 864.0
        option['initial_info'] = init_info

        image = video['video'].module('image_preprocess', option)

        (
            bmf.encode(
                image,
                None,
                {
                    "output_path": output_path,
                    "video_params": {
                        "codec": "png",
                        "width": 640,
                        "height": 480,
                        "crf": "23",
                        "preset": "veryfast"
                    }
                }
            ).run()
        )
        self.check_video_diff(output_path, expect_result)

    @timeout_decorator.timeout(seconds=120)
    def test_text_preprocess(self):
        input_video_path = "../files/jack.png"
        output_path = "./text_preprocess.png"
        expect_result = '../video_edit_prototype/text_preprocess.png|480|640|0.009000|MOV,MP4,M4A,3GP,3G2,MJ2|' \
                        '104990222|118114|png|{"fps": "120.0"}'
        self.remove_result_data(output_path)
        # create graph
        graph = bmf.graph()

        # decode
        video = graph.module('text_to_image', {
            'Text': 'jack',
            'local_path': input_video_path,
            'FontType': 'SimHei',
            'FontSize': 23
        }).decode()

        (
            bmf.encode(
                video['video'],
                None,
                {
                    "output_path": output_path,
                    "video_params": {
                        "codec": "png",
                        "width": 640,
                        "height": 480,
                        "crf": "23",
                        "preset": "veryfast"
                    }
                }
            )
                .upload()
                .run()
        )
        self.check_video_diff(output_path, expect_result)


if __name__ == '__main__':
    unittest.main()
