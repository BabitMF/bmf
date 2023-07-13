import sys
import time
from multiprocessing import Process
import multiprocessing
from threading import Thread

sys.path.append("../../")
sys.path.append("../gpu_video_frame/")

def task(proc_idx, input_video_path, frames_dict):
		# import bmf in each sub-process
    import bmf
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
    input_video_path = "../files/lark_stream0.flv"
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
	
if __name__ == "__main__":
    test_gpu_decode_multi_proc_perf()