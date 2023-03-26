# import pycuda.autoinit
import sys
import time

# import pycuda.autoinit
sys.path.append("../../..")
import bmf


def test():
    input_video_path = "../files/img.mp4"
    output_path = "./output.mp4"
    expect_result = '|1080|1920|7.615000|MOV,MP4,M4A,3GP,3G2,MJ2|4483427|4267663|h264|' \
                    '{"fps": "30.0662251656"}'
    (
        bmf.graph()
            .decode({'input_path': input_video_path})['video']
            .module('cpu_gpu_trans_module', {"to_gpu": 1})
            .module('cpu_gpu_trans_module', {"to_gpu": 0})
            .encode(None, {"output_path": output_path})
            .run()
    )


if __name__ == '__main__':
    test()
