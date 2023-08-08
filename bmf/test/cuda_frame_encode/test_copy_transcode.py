import bmf
from bmf import VideoFrame

import subprocess

input_path = "../../files/big_bunny_10s_30fps.mp4"
output_path = "output.mp4"


def check_psnr(input_file, output_file):
    cmd = 'ffmpeg  -i {} -i {} -lavfi "psnr" -f null - 2>&1 | grep Parsed_psnr'.format(
        input_path, output_path)
    psnr_text = subprocess.check_output(cmd, shell=True, text=True)
    print(psnr_text)
    for text in psnr_text.split():
        if text.startswith("min"):
            _, min_psnr = text.split(":")
            if float(min_psnr) < 40:
                raise Exception("psnr check failed")


def test_trans():
    bmf.graph().decode({
        'input_path': input_path,
        'video_params': {
            'hwaccel': 'cuda',
        },
    })['video'].module("copy_frame",
                       None,
                       entry="copy_frame.copy_frame",
                       input_manager="immediate").encode(
                           None, {
                               "video_params": {
                                   "codec": "h264_nvenc",
                                   "pix_fmt": "cuda",
                                   "vsync": "vfr",
                                   "max_fr": 30,
                               },
                               "output_path": output_path,
                           }).run()

    check_psnr(input_path, output_path)


if __name__ == "__main__":
    test_trans()
