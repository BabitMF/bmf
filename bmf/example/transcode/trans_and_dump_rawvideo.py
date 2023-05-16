import sys
import time
import unittest

sys.path.append("../../../")
import bmf

def test_encode_and_raw_data():
    input_video_path = '../files/img.mp4'
    raw_output_path = "./out.yuv"
    output_path = "./out.mp4"

    # create graph
    my_graph = bmf.graph()

    # decode video
    video1 = my_graph.decode({'input_path': input_video_path})

    # get raw data of the video stream
    raw_output = (
        bmf.encode(
            video1['video'],
            None,
            {
                "video_params": {
                    "codec": "rawvideo" # using this parameter to get raw yuv data
                },
                "format": "rawvideo",
                "output_path": raw_output_path
            }
        )
    )

    # encode
    bmf.encode(
        video1['video'],
        video1['audio'],
        {
            "video_params": {
                "codec": "h264",
                "width": 320,
                "height": 240,
                "crf": 23,
                "preset": "veryfast"
            },
            "output_path": output_path
        }
    ).run()

if __name__ == '__main__':
    test_encode_and_raw_data()
