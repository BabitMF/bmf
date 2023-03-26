import sys
import unittest

sys.path.append("../../..")
from bmf.example.video_edit_prototype.edit_task import _edit_task


class MyTestCase(unittest.TestCase):
    def test_something(self):
        video_config_info = {
            'Upload': {
                'Uploader': 'Y'
            },
            'Output': {
                'Mode': 'Normal',
                'Width': 640,
                'Height': 480,
                'Format': 'mp4',
                'SegmentTime': 10,
                'Fps': 30,
                'Quality': 'medium'
            },
            'Segments': [
                {
                    'BackGround': '0xFFFF00FF',
                    'Elements': [
                        {
                            'Type': 'video',
                            'Source': '../files/img.mp4',
                            'StartTime': 2,
                            'Duration': 3,
                            'Position': {
                                'PosX': '0%',
                                'PosY': '0%',
                                'Width': '80%',
                                'Height': '80%'
                            },
                            'Crop': {
                                'PosX': '0%',
                                'PosY': '0%',
                                'Width': '100%',
                                'Height': '100%'
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
                        },
                        {
                            'Type': 'image',
                            'Source': '../files/blue.png',
                            'StartTime': 3,
                            'Duration': 1.2,
                            'Position': {
                                'PosX': '80%',
                                'PosY': '0%',
                                'Width': '20%',
                                'Height': '20%'
                            },
                            'Crop': {
                                'PosX': '20%',
                                'PosY': '20%',
                                'Width': '80%',
                                'Height': '80%'
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
                        },
                        {
                            'Type': 'text',
                            'Text': 'Byte Dance',
                            'StartTime': 0,
                            'Duration': 5,
                            'Position': {
                                'PosX': '50%',
                                'PosY': '80%',
                                'Width': '20%',
                                'Height': '20%'
                            },
                            'FontType': 'SimHei',
                            'FontSize': 23,
                            'FontColor': '0xFFFFFF00',
                            'BackgroundColor': '0xFFFFFF00',
                            'ShadowColor': '0xFFFFFF00',
                            'HorizontalAlign': 1,
                            'VerticalAlign': 0,
                            'MultiLine': 0,
                            'LineSpace': 1.5,
                            'ReplaceSuffix': 0,
                            'Animation': {
                                'Type': 'X',
                                'Speed': 'default',
                                'Duration': 3
                            },
                            'Italic': 1,
                            'FontWeight': 'bold',
                            'Underline': 1
                        },
                        {
                            'Type': 'audio',
                            'Source': '../files/header.mp4',
                            'StartTime': 0,
                            'Duration': 1,
                            'Trims': [
                                [0, 1]
                            ],
                            'Volume': 0.5,
                            'Loop': 0
                        }
                    ]
                },
                {
                    'BackGround': '0xFFFFFF00',
                    'Elements': [
                        {
                            'Type': 'video',
                            'Source': '../files/img.mp4',
                            'StartTime': 0,
                            'Duration': 5,
                            'Position': {
                                'PosX': '20%',
                                'PosY': '20%',
                                'Width': '80%',
                                'Height': '80%'
                            },
                            'Crop': {
                                'PosX': '0%',
                                'PosY': '0%',
                                'Width': '100%',
                                'Height': '100%'
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
                            'Volume': 0.7,
                            'Mute': 0
                        },
                        {
                            'Type': 'image',
                            'Source': '../files/blue.png',
                            'StartTime': 1,
                            'Duration': 2,
                            'Position': {
                                'PosX': '0%',
                                'PosY': '0%',
                                'Width': '20%',
                                'Height': '20%'
                            },
                            'Crop': {
                                'PosX': '20%',
                                'PosY': '20%',
                                'Width': '80%',
                                'Height': '80%'
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
                        },
                        {
                            'Type': 'text',
                            'Text': 'Jin Ri Tou Tiao',
                            'StartTime': 0,
                            'Duration': 5,
                            'Position': {
                                'PosX': '50%',
                                'PosY': '0%',
                                'Width': '20%',
                                'Height': '20%'
                            },
                            'FontType': 'SimHei',
                            'FontSize': 18,
                            'FontColor': '0xFFFFFF00',
                            'BackgroundColor': '0xFFFFFF00',
                            'ShadowColor': '0xFFFFFF00',
                            'HorizontalAlign': 1,
                            'VerticalAlign': 0,
                            'MultiLine': 0,
                            'LineSpace': 1.5,
                            'ReplaceSuffix': 0,
                            'Animation': {
                                'Type': 'X',
                                'Speed': 'default',
                                'Duration': 3
                            },
                            'Italic': 1,
                            'FontWeight': 'bold',
                            'Underline': 1
                        },
                        {
                            'Type': 'audio',
                            'Source': '../files/header.mp4',
                            'StartTime': 0,
                            'Duration': 1,
                            'Trims': [
                                [0, 1]
                            ],
                            'Volume': 2,
                            'Loop': 0
                        }
                    ]
                }
            ],
            'GlobalElements': [
                {
                    'Type': 'text',
                    'Text': 'Always Day 1~',
                    'StartTime': 0,
                    'Duration': 10,
                    'Position': {
                        'PosX': '0%',
                        'PosY': '80%',
                        'Width': '20%',
                        'Height': '20%'
                    },
                    'FontType': 'SimHei',
                    'FontSize': 15,
                    'FontColor': '0xFFFFFF00',
                    'BackgroundColor': '0xFFFFFF00',
                    'ShadowColor': '0xFFFFFF00',
                    'HorizontalAlign': 1,
                    'VerticalAlign': 0,
                    'MultiLine': 0,
                    'LineSpace': 1.5,
                    'ReplaceSuffix': 0,
                    'Animation': {
                        'Type': 'X',
                        'Speed': 'default',
                        'Duration': 3
                    },
                    'Italic': 1,
                    'FontWeight': 'bold',
                    'Underline': 1
                },
                {
                    'Type': 'image',
                    'Source': '../files/overlay.png',
                    'StartTime': 0,
                    'Duration': 10,
                    'Position': {
                        'PosX': '80%',
                        'PosY': '80%',
                        'Width': '20%',
                        'Height': '20%'
                    },
                    'Crop': {
                        'PosX': '0%',
                        'PosY': '0%',
                        'Width': '100%',
                        'Height': '100%'
                    },
                    'Rotate': 0,
                    "Filters": {
                        "Contrast": 110,
                        "Brightness": 62,
                        "Saturate": 121,
                        "Opacity": 66,
                        "Blur": 28
                    }
                },
                {
                    'Type': 'audio',
                    'Source': '../files/header.mp4',
                    'StartTime': 4,
                    'Duration': 1,
                    'Trims': [
                        [0, 1]
                    ],
                    'Volume': 0.5,
                    'Loop': 0
                },
                {
                    'Type': 'video',
                    'Source': '../files/xigua_loop_logo2_x.mov',
                    'StartTime': 0,
                    'Duration': 10,
                    'Position': {
                        'PosX': '50%',
                        'PosY': '50%',
                        'Width': '10%',
                        'Height': '10%'
                    },
                    'Crop': {
                        'PosX': '0%',
                        'PosY': '0%',
                        'Width': '100%',
                        'Height': '100%'
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
            ]
        }
        result_file = _edit_task(upload=video_config_info['Upload'], output=video_config_info['Output'],
                                 segments=video_config_info['Segments'],
                                 global_elements=video_config_info['GlobalElements'])
        print ('result file : ' + str(result_file))


if __name__ == '__main__':
    unittest.main()
