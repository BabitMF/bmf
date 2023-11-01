import sys

sys.path.append("../../")
import bmf

sys.path.pop()

def test():
    input_video_path = "./MAT/test_sets/CelebA-HQ/images/test1.png"
    output_path = "./output.mp4"

    graph = bmf.graph()
    video = graph.decode({
        "input_path": input_video_path,
        # "video_params": {
        #     "hwaccel": "cuda",
        #     # "pix_fmt": "yuv420p",
        # }
    })
    (video['video'].module('inpaint_module'
    ).encode(
        None, {
            "output_path": output_path,
            "video_params": {
                "codec": "hevc_nvenc",
                "pix_fmt": "cuda",
            }
        }).run())


if __name__ == '__main__':
    test()
