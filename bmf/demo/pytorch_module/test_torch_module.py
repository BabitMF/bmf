import sys

sys.path.append("../../")
import bmf

sys.path.pop()

def test():
    input_video_path = "./test1.png"
    output_path = "./output.mp4"

    graph = bmf.graph()
    video = graph.decode({
        "input_path": input_video_path,
        # "video_params": {
        #     "hwaccel": "cuda",
        #     # "pix_fmt": "yuv420p",
        # }
    })
    (video['video']
     .module('torch_padding_module', {
        'padding': '10,10'
    })
    .encode(
        None, {
            "output_path": output_path,
            "video_params": {
                "codec": "hevc_nvenc",
                "pix_fmt": "cuda",
            }
        }).run())


if __name__ == '__main__':
    test()
