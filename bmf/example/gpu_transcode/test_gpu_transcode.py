import sys
import time
from multiprocessing import Process
import multiprocessing
from threading import Thread

sys.path.append("../../")
sys.path.append("../gpu_video_frame/")
sys.path.append("../gpu_module/")
import bmf

# ===================== decode examples =======================

def test_gpu_decode():
    print("Testing gpu decoding......")
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

def decode_task(proc_idx, input_video_path, frames_dict):
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

def test_gpu_decode_multi_thread_perf():
    print("Testing gpu decoding using multiple threads.....")
    input_video_path = "../files/lark_stream0.flv"
    num_threads = 2
    threads = []
    frames_dict = dict()

    for i in range(num_threads):
        threads.append(Thread(target=decode_task, args=(i, input_video_path, frames_dict)))
    
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

# ===================== decode examples =======================

def test_gpu_encode():
    print("Testing gpu encoding......")
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

def encode_task(num_frames):
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
                    },
                    "vframes": num_frames,
                    "format": "null",
                }
           ).run()
    )

def test_gpu_encode_multi_thread_perf():
    print("Testing gpu encoding using multiple threads......")
    num_threads = 2
    num_frames = 100
    threads = []

    for i in range(num_threads):
        threads.append(Thread(target=encode_task, args=(num_frames,)))

    start = time.time()
    for i in range(num_threads):
        threads[i].start()
    
    for i in range(num_threads):
        threads[i].join()
    stop = time.time()
    
    duration = stop - start
    total_frames = num_threads * num_frames

    print("Total Frames Encoded={}, time={} seconds, FPS under multiple threads={}".format(total_frames, duration, total_frames / duration))

# ===================== decode examples =======================

def test_gpu_transcode():
    print("Testing gpu transcoding......")
    input_video_path = "../files/lark_stream0.flv"
    output_video_path = "./gpu_transcoded_result.mp4"

    graph = bmf.graph()

    video = graph.decode({"input_path": input_video_path,
                          "video_params": {
                              "hwaccel": "cuda",
                           }})
    (
        bmf.encode(
            video["video"].module("scale_gpu", {"size": "1280x720"}),
            video["audio"],
            {
                "output_path": output_video_path,
                "video_params": {
                    "codec": "h264_nvenc",
                    "pix_fmt": "cuda",
                    "bit_rate": 5000000,
                },
            }
        ).run()
    )

def test_gpu_transcode_1_to_n():
    print("Testing gpu 1-n transcoding......")
    input_video_path = "../files/lark_stream0.flv"
    output_video_path_1 = "./gpu_transcoded_result_1.mp4"
    output_video_path_2 = "./gpu_transcoded_result_2.mp4"

    graph = bmf.graph()

    video = graph.decode({"input_path": input_video_path,
                          "video_params": {
                              "hwaccel": "cuda",
                           }})
    bmf.encode(
        video["video"].module("scale_gpu", {"size": "1280x720"}),
        video["audio"],
        {
            "output_path": output_video_path_1,
            "video_params": {
                "codec": "h264_nvenc",
                "pix_fmt": "cuda",
                "bit_rate": 5000000,
            },
        }
    )

    bmf.encode(
        video["video"].module("scale_gpu", {"size": "960x540"}),
        video["audio"],
        {
            "output_path": output_video_path_2,
            "video_params": {
                "codec": "h264_nvenc",
                "pix_fmt": "cuda",
                "bit_rate": 5000000,
            },
        }
    )   

    graph.run()

def transcode_task():
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
    print("Testing gpu transcoding using multiple threads......")
    input_video_path = "../files/lark_stream0.flv"
    num_threads = 2
    threads = []

    num_frames = get_total_frames(input_video_path)

    for i in range(num_threads):
        threads.append(Thread(target=transcode_task))
    
    start = time.time()
    for i in range(num_threads):
        threads[i].start()
    
    for i in range(num_threads):
        threads[i].join()
    stop = time.time()

    duration = stop - start
    total_frames = num_threads * num_frames
    
    print("Total Frames Transcoded={}, time={} seconds, FPS under multiple threads={}".format(total_frames, duration, total_frames / duration))

# ===================== gpu transcoding with cuda filter examples =======================

def test_gpu_transcode_with_scale_cuda():
    print("Testing gpu transcoding with scale cuda filter......")
    input_video_path = "../files/lark_stream0.flv"
    output_video_path = "./gpu_trans_scale_cuda.mp4"

    graph = bmf.graph()

    video = graph.decode({"input_path": input_video_path,
                          "video_params": {
                              "hwaccel": "cuda",
                          }})
    (
        bmf.encode(
            video["video"].ff_filter("scale_cuda", w=1280, h=720),
            video["audio"],
            {
                "output_path": output_video_path,
                "video_params": {
                    "codec": "h264_nvenc",
                    "pix_fmt": "cuda",
                    "bit_rate": 5000000,
                },
            }).run()
    )

def test_gpu_transcode_with_hwupload():
    print("Testing gpu transcoding with hwupload filter......")
    input_video_path = "../files/lark_stream0.flv"
    output_video_path = "./gpu_trans_hwupload.mp4"

    graph = bmf.graph()

    video = graph.decode({"input_path": input_video_path})
    (
        bmf.encode(
            video["video"].ff_filter("hwupload_cuda")
                          .ff_filter("scale_cuda", w=1280, h=720),
            video["audio"],
            {
                "output_path": output_video_path,
                "video_params": {
                    "codec": "h264_nvenc",
                    "pix_fmt": "cuda",
                    "bit_rate": 5000000,
                },
            }).run()
    )

def test_gpu_transcode_with_scale_npp():
    print("Testing gpu transcoding with scale npp filter......")
    input_video_path = "../files/lark_stream0.flv"
    output_video_path = "./gpu_trans_scale_npp.mp4"

    graph = bmf.graph()

    video = graph.decode({"input_path": input_video_path,
                          "video_params": {
                              "hwaccel": "cuda",
                          }})
    (
        bmf.encode(
            video["video"].ff_filter("scale_npp", w=1280, h=720),
            video["audio"],
            {
                "output_path": output_video_path,
                "video_params": {
                    "codec": "h264_nvenc",
                    "pix_fmt": "cuda",
                    "bit_rate": 5000000,
                },
            }).run()
    )

def test_gpu_transcode_with_yadif_cuda():
    print("Testing gpu transcoding with yadif filter......")
    input_video_path = "../files/lark_stream0.flv"
    output_video_path = "./gpu_trans_yadif_cuda.mp4"

    graph = bmf.graph()

    video = graph.decode({"input_path": input_video_path,
                          "video_params": {
                              "hwaccel": "cuda",
                          }})
    (
        bmf.encode(
            video["video"].ff_filter("scale_cuda", w=1280, h=720)
                          .ff_filter("yadif_cuda"),
            video["audio"],
            {
                "output_path": output_video_path,
                "video_params": {
                    "codec": "h264_nvenc",
                    "pix_fmt": "cuda",
                    "bit_rate": 5000000,
                },
            }).run()
    )   

def test_gpu_transcode_with_overlay_cuda():
    print("Testing gpu transcoding with overlay filter......")
    input_video_path_1 = "../files/lark_stream0.flv"
    input_video_path_2 = "../files/nvidia-logo.png"
    output_video_path = "./gpu_trans_with_overlay.mp4"

    graph = bmf.graph()

    video1 = (
        graph.decode({"input_path": input_video_path_1,
                          "video_params": {
                              "hwaccel": "cuda"
                         }
                     })["video"]
             .ff_filter("scale_npp", format="yuv420p")
    )

    logo = (
        graph.decode({"input_path": input_video_path_2})["video"]
             .ff_filter("scale", "180:-2")
             .ff_filter("format", "yuva420p")
             .ff_filter("hwupload_cuda")
    )

    (
        bmf.encode(
            bmf.ff_filter([video1, logo], 'overlay_cuda', x=50, y=50),
            None,
            {
                "output_path": output_video_path,
                "video_params": {
                    "codec": "h264_nvenc",
                    "pix_fmt": "cuda",
                    "bit_rate": 5000000,
                },
            }).run()
    )

    

if __name__ == "__main__":
    test_gpu_decode()
    test_gpu_decode_multi_thread_perf()
    test_gpu_encode()
    test_gpu_encode_multi_thread_perf()
    test_gpu_transcode()
    test_gpu_transcode_1_to_n()
    test_gpu_transcode_multi_thread_perf()
    test_gpu_transcode_with_scale_cuda()
    test_gpu_transcode_with_hwupload()
    test_gpu_transcode_with_scale_npp()
    test_gpu_transcode_with_yadif_cuda()
    test_gpu_transcode_with_overlay_cuda()
