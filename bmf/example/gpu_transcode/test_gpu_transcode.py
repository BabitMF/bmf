import sys
import time
from multiprocessing import Process
import multiprocessing
from threading import Thread

sys.path.append("../../")
sys.path.append("../gpu_video_frame/")
sys.path.append("../gpu_module/")
import bmf

def test_gpu_transcode():
    input_video_path = "../files/lark_stream0.flv"
    output_video_path = "./gpu_transcoded_result.mp4"

    graph = bmf.graph()

    video = graph.decode({"input_path": input_video_path,
                          "video_params": {
                              "hwaccel": "cuda",
                           }})
    (
        bmf.encode(
            video["video"].module("scale_cvcuda", {"size": "1280x720"}),
            video["audio"],
            {
                "output_path": output_video_path,
                "video_params": {
                    "codec": "h264_nvenc",
                    #"pix_fmt": "cuda",
                    "preset": "p6",
                    "tune": "hq",
                    "bit_rate": 5000000,
                },
            }
        ).run()
    )

def test_gpu_transcode_1_to_n():
    input_video_path = "../files/lark_stream0.flv"
    output_video_path_1 = "./gpu_transcoded_result_1.mp4"
    output_video_path_2 = "./gpu_transcoded_result_2.mp4"

    graph = bmf.graph()

    video = graph.decode({"input_path": input_video_path,
                          "video_params": {
                              "hwaccel": "cuda",
                           }})
    bmf.encode(
        video["video"].module("scale_cvcuda", {"size": "1280x720"}),
        video["audio"],
        {
            "output_path": output_video_path_1,
            "video_params": {
                "codec": "h264_nvenc",
                #"pix_fmt": "cuda",
                "preset": "p6",
                "tune": "hq",
                "bit_rate": 5000000,
            },
        }
    )

    bmf.encode(
        video["video"].module("scale_cvcuda", {"size": "960x540"}),
        video["audio"],
        {
            "output_path": output_video_path_2,
            "video_params": {
                "codec": "h264_nvenc",
                #"pix_fmt": "cuda",
                "preset": "p4",
                "tune": "hq",
                "bit_rate": 5000000,
            },
        }
    )   

    graph.run()

def task():
    input_video_path = "../files/lark_stream0.flv"
    graph = bmf.graph()

    video = graph.decode({"input_path": input_video_path,
                          "video_params": {
                              "hwaccel": "cuda",
                           }})
    (
        bmf.encode(
            video["video"],
            video["audio"],
            {
                "video_params": {
                    "codec": "h264_nvenc",
                    "pix_fmt": "cuda",
                    "preset": "p3",
                    "tune": "hq"
                },
                "format": "null",
            }
        ).run()
    )

def get_total_frames(input_video_path):
    graph = bmf.graph()

    frames = graph.decode({
        "input_path": input_video_path,
        "video_params": {
            "hwaccel": "cuda",
        }})["video"].start()
    
    num_frames = 0
    for i in frames:
        num_frames += 1
    
    return num_frames

def test_gpu_transcode_multi_thread_perf():
    input_video_path = "../files/lark_stream0.flv"
    num_threads = 2
    threads = []

    num_frames = get_total_frames(input_video_path)

    for i in range(num_threads):
        threads.append(Thread(target=task))
    
    start = time.time()
    for i in range(num_threads):
        threads[i].start()
    
    for i in range(num_threads):
        threads[i].join()
    stop = time.time()

    duration = stop - start
    total_frames = num_threads * num_frames
    
    print("Total Frames Transcoded={}, time={} seconds, FPS under multiple threads={}".format(total_frames, duration, total_frames / duration))

if __name__ == "__main__":
    test_gpu_transcode()
    test_gpu_transcode_1_to_n()
    test_gpu_transcode_multi_thread_perf()