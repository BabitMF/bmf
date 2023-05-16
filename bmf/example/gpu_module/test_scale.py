import sys
import time

# import pycuda.autoinit
sys.path.append("../../")
import bmf


def test():
    input_video_path = "../files/only_video.mp4"
    output_path = "./output.mp4"
    # expect_result = '|1080|1920|7.615000|MOV,MP4,M4A,3GP,3G2,MJ2|4483427|4267663|h264|' \
    #                 '{"fps": "30.0662251656"}'

    graph = bmf.graph()
    video = graph.decode({
        "input_path": input_video_path,
        "video_params": {
            "hwaccel": "cuda",
            "extract_frames": {
                "device": "cuda"
            }
        }
    })
    (
        video['video']
            .module('scale_gpu', {"size": '720x1280', 'algo': 'cubic'})
            # .module('flip_gpu', {'direction': 'v'})
            # .module('rotate_gpu', {'angle': 'pi/8'})
            # .module('crop_gpu', {'x': 0, 'y': 0, 'width': 480, 'height': 640})
            # .module('blur_gpu', {'op': 'gblur', 'sigma': [0.7, 0.7], 'size': [5, 5]})
            .encode(None, {
                "output_path": output_path,
                "video_params": {
                    "codec": "hevc_nvenc",
                    "pix_fmt": "cuda",
                    # "width": 720,
                    # "height": 1280,
                }
            })
            .run()
    )


if __name__ == '__main__':
    test()
