import sys
import time
from multiprocessing import Process
import multiprocessing
from threading import Thread

sys.path.append("../../")
sys.path.append("../gpu_video_frame/")
import bmf

def test_gpu_decode():
    input_video_path = "../files/lark_stream0.flv"
    output_path = "./gpu_decode_result.yuv"

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
                    "output_path": output_path
                }
            ).run()
    )

def task(proc_idx, input_video_path, frames_dict):
    graph = bmf.graph()

    frames = graph.decode({
        "input_path": input_video_path,
        "video_params": {
            "hwaccel": "cuda",
        }})["video"].start()
    
    num_frames = 0
    for i in frames:
        num_frames += 1

    frames_dict[str(proc_idx)] = num_frames

def test_gpu_decode_multi_proc_perf():
    input_video_path = "../files/perf_1080p_10k.h264"
    num_processes = 2
    processes = []
    frames_dict = multiprocessing.Manager().dict()

    for i in range(num_processes):
        processes.append(Process(target=task, args=(i, input_video_path, frames_dict)))
    
    start = time.time()
    for i in range(num_processes):
        processes[i].start()

    for i in range(num_processes):
        processes[i].join()
    stop = time.time()

    duration = stop - start
    total_frames = 0
    for i in range(num_processes):
        total_frames += frames_dict[str(i)]

    print("Total Frames Decoded={}, time={} seconds, FPS under multiple processes={}".format(total_frames, duration, total_frames / duration))

def test_gpu_decode_multi_thread_perf():
    input_video_path = "../files/perf_1080p_10k.h264"
    num_threads = 2
    threads = []
    frames_dict = dict()

    for i in range(num_threads):
        threads.append(Thread(target=task, args=(i, input_video_path, frames_dict)))
    
    start = time.time()
    for i in range(num_threads):
        threads[i].start()

    for i in range(num_threads):
        threads[i].join()
    stop = time.time()

    duration = stop - start
    total_frames = 0
    for i in range(num_threads):
        total_frames += frames_dict[str(i)]

    print("Total Frames Decoded={}, time={} seconds, FPS under multiple threads={}".format(total_frames, duration, total_frames / duration))



if __name__ == "__main__":
    test_gpu_decode()
    test_gpu_decode_multi_thread_perf()