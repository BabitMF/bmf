import sys
import time
import unittest

sys.path.append("../../")
import bmf
import timeout_decorator
import threading
import os

sys.path.append("../../bmf/example/")
from base_test.base_test_case import BaseTestCase
from base_test.media_info import MediaInfo


def jobv(index):
    input = "../files/edit" + str(index) + ".mp4"
    output = "../files/vv_" + str(index) + ".mp4"
    cmd = "ffmpeg -y -i " + input + " -t 10.0 -filter_complex " \
                                    "'[0:v]scale=-2:720,setsar=r=1/1[s],[s]pad=w=1280:h=720:x=(ow-iw)/2:y=(oh-ih)/2:color=black,setsar=r=1/1' " \
                                    "-c:v libx264 -crf 23 -preset veryfast -r 20.0 -an -y " + output
    os.system(cmd)


def longvideo_jobv(index):
    input = "../files/edit2.mp4"
    output = "../files/vv_" + str(index) + ".mp4"
    cmd = "ffmpeg -y -i " + input + " -t 60.0 -filter_complex " \
                                    "'[0:v]scale=-2:720,setsar=r=1/1[s],[s]pad=w=1280:h=720:x=(ow-iw)/2:y=(oh-ih)/2:color=black,setsar=r=1/1' " \
                                    "-c:v libx264 -crf 23 -preset veryfast -r 20.0 -an -y " + output
    os.system(cmd)


class TestCompareWithEdit(BaseTestCase):
    @timeout_decorator.timeout(seconds=120)
    def test_edit_concat(self):

        input_video_path_1 = "../files/edit1.mp4"
        input_video_path_2 = "../files/edit2.mp4"
        input_video_path_3 = "../files/edit3.mp4"
        output_path = "./edit_concat.mp4"
        expect_result = '../../../test/case/edit_concat.mp4|720|1280|29.959000|MOV,MP4,M4A,3GP,3G2,MJ2|1243904|' \
                        '4664643|h264|{"fps": "20.0278164117"}'
        self.remove_result_data(output_path)

        concat_option = {
            "dump_graph": 0,
            "width": 1280,
            "height": 720,
            # if have audio input
            "has_audio": 0,
            "video_list": [
                {
                    "start": 0,
                    "duration": 10,
                    "transition_time": 1,
                    "transition_mode": 1
                },
                {
                    "start": 0,
                    "duration": 10,
                    "transition_time": 1,
                    "transition_mode": 1
                }
            ]
        }
        concat_option2 = {
            "dump_graph": 0,
            "width": 1280,
            "height": 720,
            # if have audio input
            "has_audio": 0,
            "video_list": [
                {
                    "start": 0,
                    "duration": 20,
                    "transition_time": 1,
                    "transition_mode": 1
                },
                {
                    "start": 0,
                    "duration": 10,
                    "transition_time": 1,
                    "transition_mode": 1
                }
            ]
        }

        # create graph
        my_graph = bmf.graph({
            "dump_graph": 0
        })

        # three videos
        video1 = my_graph.decode({'input_path': input_video_path_1})
        video2 = my_graph.decode({'input_path': input_video_path_2})
        video3 = my_graph.decode({'input_path': input_video_path_3})

        v1 = video1['video']
        v2 = video2['video']
        v3 = video3['video']

        # do video_norm
        v1 = bmf.module([v1], 'video_norm')
        v2 = bmf.module([v2], 'video_norm')
        v3 = bmf.module([v3], 'video_norm')

        # do concat1
        concat_streams = (
            bmf.module([
                v1,
                v2
            ], 'video_concat2', concat_option)
        )

        # do concat2
        concat_streams = (
            bmf.module([
                concat_streams[0],
                v3
            ], 'video_concat2', concat_option2)
        )

        # encode
        (
            bmf.encode(concat_streams[0], None,
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
    def test_edit_concat_ffmpeg(self):

        input_video_path_1 = "../files/edit1.mp4"
        input_video_path_2 = "../files/edit2.mp4"
        input_video_path_3 = "../files/edit3.mp4"
        output_path = "./edit_concat_ffmpeg.mp4"
        expect_result = '../../../test/case/edit_concat_ffmpeg.mp4|720|1280|30.000000|MOV,MP4,M4A,3GP,3G2,MJ2|' \
                        '798864|2995741|h264|{"fps": "20.0"}'
        self.remove_result_data(output_path)
        self.set_ffmpeg_env()

        ts = []
        for i in range(1, 4):
            t = threading.Thread(target=jobv, args=(i,))
            ts.append(t)

        for i in ts:
            i.start()
        for i in ts:
            i.join()

        cmd4 = "ffmpeg -y -i ../files/vv_1.mp4 -i ../files/vv_2.mp4 -i ../files/vv_3.mp4 -filter_complex " \
               "'[0:v]scale=1280:720[v1];[v1]split[sp1][sp2];[sp1]trim=start=0:duration=10[v2];[v2]setpts=PTS-STARTPTS[v3];[sp2]trim=start=9:duration=1[v4];" \
               "[v4]setpts=PTS-STARTPTS[v5];[v5]scale=200:200[v6];[1:v]scale=1280:720[v7];[v7]split[sp3][sp4];[sp3]trim=start=0:duration=10[v8];" \
               "[v8]setpts=PTS-STARTPTS[v9];[v9][v6]overlay=repeatlast=0[v10];[sp4]trim=start=9:duration=1[v11];" \
               "[v11]setpts=PTS-STARTPTS[v12];[v12]scale=200:200[v13];[2:v]scale=1280:720[v14];[v14]trim=start=0:duration=10[v15];" \
               "[v15]setpts=PTS-STARTPTS[v16];[v16][v13]overlay=repeatlast=0[v17];[v3][v10][v17]concat=n=3:v=1:a=0[v18]" \
               "' -map '[v18]' -c:v libx264 -crf 23 -r 20 -s 1280x720 -preset veryfast %s" % (output_path)

        os.system(cmd4)

        self.check_video_diff(output_path, expect_result)

    @timeout_decorator.timeout(seconds=120)
    def test_longvideo_edit_concat(self):

        input_video_path_1 = "../files/edit2.mp4"
        input_video_path_2 = "../files/edit2.mp4"
        input_video_path_3 = "../files/edit2.mp4"
        output_path = "./longvideo_edit_concat.mp4"
        expect_result = '../../../test/case/longvideo_edit_concat.mp4|720|1280|30.0|MOV,MP4,M4A,3GP,3G2,MJ2|' \
                        '1797288|7222707|h264|{"fps": "20.0046307016"}'
        self.remove_result_data(output_path)

        concat_option = {
            "dump_graph": 0,
            "width": 1280,
            "height": 720,
            # if have audio input
            "has_audio": 0,
            "video_list": [
                {
                    "start": 0,
                    "duration": 60,
                    "transition_time": 5,
                    "transition_mode": 1
                },
                {
                    "start": 0,
                    "duration": 60,
                    "transition_time": 5,
                    "transition_mode": 1
                }
            ]
        }
        concat_option2 = {
            "dump_graph": 0,
            "width": 1280,
            "height": 720,
            # if have audio input
            "has_audio": 0,
            "video_list": [
                {
                    "start": 0,
                    "duration": 120,
                    "transition_time": 5,
                    "transition_mode": 1
                },
                {
                    "start": 0,
                    "duration": 60,
                    "transition_time": 5,
                    "transition_mode": 1
                }
            ]
        }

        # create graph
        my_graph = bmf.graph({
            "dump_graph": 0
        })

        # three videos
        video1 = my_graph.decode({'input_path': input_video_path_1})
        video2 = my_graph.decode({'input_path': input_video_path_2})
        video3 = my_graph.decode({'input_path': input_video_path_3})

        v1 = video1['video']
        v2 = video2['video']
        v3 = video3['video']

        # do video_norm
        v1 = bmf.module([v1], 'video_norm')
        v2 = bmf.module([v2], 'video_norm')
        v3 = bmf.module([v3], 'video_norm')

        # do concat1
        concat_streams = (
            bmf.module([
                v1,
                v2
            ], 'video_concat2', concat_option)
        )

        # do concat2
        concat_streams = (
            bmf.module([
                concat_streams[0],
                v3
            ], 'video_concat2', concat_option2)
        )

        # encode
        (
            bmf.encode(concat_streams[0], None,
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
    def test_longvideo_edit_concat_ffmpeg(self):

        input_video_path_1 = "../files/edit2.mp4"
        input_video_path_2 = "../files/edit2.mp4"
        input_video_path_3 = "../files/edit2.mp4"
        output_path = "./longvideo_edit_concat_ffmpeg.mp4"
        expect_result = '../../../test/case/longvideo_edit_concat_ffmpeg.mp4|720|1280|30.3|' \
                        'MOV,MP4,M4A,3GP,3G2,MJ2|1436543|4697932|h264|{"fps": "20.0"}'
        self.remove_result_data(output_path)
        self.set_ffmpeg_env()

        ts = []
        for i in range(1, 4):
            t = threading.Thread(target=longvideo_jobv, args=(i,))
            ts.append(t)

        for i in ts:
            i.start()
        for i in ts:
            i.join()

        cmd4 = "ffmpeg -y -i ../files/vv_1.mp4 -i ../files/vv_2.mp4 -i ../files/vv_3.mp4 -filter_complex " \
               "'[0:v]scale=1280:720[v1];[v1]split[sp1][sp2];[sp1]trim=start=0:duration=60[v2];[v2]setpts=PTS-STARTPTS[v3];[sp2]trim=start=55:duration=5[v4];" \
               "[v4]setpts=PTS-STARTPTS[v5];[v5]scale=200:200[v6];[1:v]scale=1280:720[v7];[v7]split[sp3][sp4];[sp3]trim=start=0:duration=60[v8];" \
               "[v8]setpts=PTS-STARTPTS[v9];[v9][v6]overlay=repeatlast=0[v10];[sp4]trim=start=55:duration=5[v11];" \
               "[v11]setpts=PTS-STARTPTS[v12];[v12]scale=200:200[v13];[2:v]scale=1280:720[v14];[v14]trim=start=0:duration=60[v15];" \
               "[v15]setpts=PTS-STARTPTS[v16];[v16][v13]overlay=repeatlast=0[v17];[v3][v10][v17]concat=n=3:v=1:a=0[v18]" \
               "' -map '[v18]' -c:v libx264 -crf 23 -r 20 -s 1280x720 -preset veryfast %s" % (output_path)

        os.system(cmd4)

        self.check_video_diff(output_path, expect_result)


if __name__ == '__main__':
    unittest.main()
