import sys
import time
from multiprocessing import Process
import multiprocessing
from threading import Thread

sys.path.append("../../")
sys.path.append("../gpu_video_frame/")
import bmf

def test_gpu_encode():
    input_video_path = "../files/lark_stream0.flv"
    yuv_path = "./gpu_decode_result.yuv"
    output_path = "gpu_encode_result.h264"

    (
        bmf.graph()
            .decode({"input_path": input_video_path,
                    "video_params": {
                        "hwaccel": "cuda",
                    }})["video"]
           .module("cpu_gpu_trans_module", {"to_gpu": 0})
           .encode(
                None,
                {
                    "video_params": {
                        "codec": "rawvideo",
                    },
                    "format": "rawvideo",
                    "output_path": yuv_path
                }
            ).run()
    )

    (
        bmf.graph()
           .decode({"input_path": yuv_path,
                    "video_params": {
                        "vsync": 0,
                    },
                    "pix_fmt": "yuv420p",
                    "s": "1920x1080"
                    })["video"]
           .encode(
                None,
                {
                    "video_params": {
                        "codec": "h264_nvenc",
                    },
                    "output_path": output_path
                }
           ).run()
    )
    """
        bmf.graph()
           .decode({"input_path": yuv_path,
                    "video_params": {
                        "vsync": 0,
                    },
                    "pix_fmt": "yuv420p",
                    "s": "1920x1080"
                    })["video"]
           .module("cpu_gpu_trans_module", {"to_gpu": 1})
           .encode(
                None,
                {
                    "video_params": {
                        "codec": "h264_nvenc",
                        "pix_fmt": "cuda",
                    },
                    "output_path": output_path
                }
           ).run()
        """

def task(num_frames):
    yuv_path = "./gpu_decode_result.yuv"

    (
        bmf.graph()
            .decode({"input_path": yuv_path,
                    "video_params": {
                        "vsync": 0,
                    },
                    "pix_fmt": "yuv420p",
                    "s": "1920x1080"
                    })["video"]
            .encode(
                None,
                {
                    "video_params": {
                        "codec": "h264_nvenc",
                        "frames": num_frames,
                    },
                    "format": "null",
                }
           ).run()
    )

def test_gpu_encode_multi_thread_perf():
    num_threads = 2
    num_frames = 100
    threads = []

    for i in range(num_threads):
        threads.append(Thread(target=task, args=(num_frames,)))

    start = time.time()
    for i in range(num_threads):
        threads[i].start()
    
    for i in range(num_threads):
        threads[i].join()
    stop = time.time()
    
    duration = stop - start
    total_frames = num_threads * num_frames

    print("Total Frames Encoded={}, time={} seconds, FPS under multiple threads={}".format(total_frames, duration, total_frames / duration))

if __name__ == "__main__":
    test_gpu_encode()
    test_gpu_encode_multi_thread_perf()