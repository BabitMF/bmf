import sys
import time

# import pycuda.autoinit
# sys.path.append("../../")
import bmf


def test():
    input_video_path = "../files/demo_JHH.mp4"
    output_path = "./output.mp4"

    graph = bmf.graph()
    video = graph.decode({
        "input_path": input_video_path,
        "video_params": {
            "hwaccel": "cuda",
            # "pix_fmt": "yuv420p",
        }
    })
    (
        video['video']
            .module('scale_gpu', {"size": '1920x1080', 'algo': 'cubic'})
            .module('flip_gpu', {'direction': 'v'})
            .module('rotate_gpu', {'angle': 'pi/8'})
            .module('crop_gpu', {'x': 0, 'y': 0, 'width': 480, 'height': 640})
            .module('blur_gpu', {'op': 'gblur', 'sigma': [0.7, 0.7], 'size': [5, 5]})
            .encode(None, {
                "output_path": output_path,
                "video_params": {
                    "codec": "hevc_nvenc",
                    "pix_fmt": "cuda",
                }
            })
            .run()
    )


if __name__ == '__main__':
    test()
