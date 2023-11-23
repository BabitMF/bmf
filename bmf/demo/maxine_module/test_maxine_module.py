import sys
import time
from threading import Thread

sys.path.append("../../")

import bmf

def test_maxine_effect(effect_options, output_file):
    input_video_path = "1080p.flv"
    output_path = output_file

    module_path = "./build/libmaxine_module.so"
    module_entry = "maxine_module:MaxineModule"

    maxine_mod = bmf.create_module({
        "name": "maxine_module",
        "path": module_path,
        "entry": module_entry,
    }, option=effect_options)

    video = bmf.graph().decode({
        "input_path": input_video_path,
        "video_params": {
            "hwaccel": "cuda",
        }
    })["video"]
    
    video2 = video.module("maxine_module", pre_module=maxine_mod)
    
    (bmf.encode(
        video2,
        video["audio"],
        {
            "output_path": output_path,
            "video_params": {
                "codec": "h264_nvenc",
                "pix_fmt": "cuda",
            }
        }).run())
        
def effect_task(thread_idx, effect_options, input_video_path, output_file):
    output_path = str(thread_idx) + output_file

    module_path = "./build/libmaxine_module.so"
    module_entry = "maxine_module:MaxineModule"

    video = bmf.graph().decode({
        "input_path": input_video_path,
        "video_params": {
            "hwaccel": "cuda",
        }
    })["video"]
    
    video2 = video.c_module('maxine_module',
                   module_path,
                   module_entry,
                   option=effect_options)
    
    (bmf.encode(
        video2,
        video["audio"],
        {
            "output_path": output_path,
            "video_params": {
                "codec": "h264_nvenc",
                "pix_fmt": "cuda",
            }
        }).run())

def test_effect_perf(effect_options, input_video_path, output_file, num_threads):

    graph = bmf.graph()

    frames = graph.decode({
        "input_path": input_video_path,
        "video_params": {
            "hwaccel": "cuda",
        }
    })["video"].start()

    num_frames = 0
    for i in frames:
        num_frames += 1
    total_frames = num_frames * num_threads

    threads = []
    for i in range(num_threads):
        threads.append(
            Thread(target=effect_task,
                   args=(i, effect_options, input_video_path, output_file))
        )
    
    start = time.time()
    for i in range(num_threads):
        threads[i].start()

    for i in range(num_threads):
        threads[i].join()
    stop = time.time()

    duration = stop - start
    print(
        "Total Frames Effected={}, time={} seconds, FPS under multiple threads={}"
        .format(total_frames, duration, total_frames / duration))

def main():
    model_dir = sys.argv[1]
    """
    Resolution Support for input videos
    Input Resolution Change        Output resolution range
    [90p, 1080p]                   [90p, 1080p]
    """
    artifact_reduction_options = {
        "num_effects": 1,
        "model_dir": model_dir,
        "ArtifactReduction": {
            "mode": 1,
            "index": 0,
        }
    }

    """
    Scale and Resolution Support for Input Videos
    Scale     input resolution change     output resolution range
    4/3x      [90p, 2160p]                [120p, 2880p]
    1.5x      [90p, 2160p]                [135p, 3240p]
    2x        [90p, 2160p]                [180p, 4320p]
    3x        [90p, 720p]                 [270p, 2160p]
    4x        [90p, 540p]                 [360p, 2160p]
    """
    super_res_options = {
        "num_effects": 1,
        "model_dir": model_dir,
        "SuperRes": {
            "resolution": 2160, # the height of results
            "mode": 0,
            "index": 0,
        }
    }

    """
    Upscale supports any input resolution and can be upscaled 4/3x, 1.5x, 2x, 3x, or 4x.
    """
    upscale_options = {
        "num_effects": 1,
        "model_dir": model_dir,
        "Upscale": {
            "resolution": 2160,
            "index": 0,
        }
    }

    artifact_reduction_upscale_options = {
        "num_effects": 2,
        "model_dir": model_dir,
        "ArtifactReduction": {
            "mode": 1,
            "index": 0,
        },
        "Upscale": {
            "resolution": 2160,
            "index": 1,
        },
    }

    green_screen_options = {
        "num_effects": 1,
        "model_dir": model_dir,
        "GreenScreen": {
            "comp_mode": 2, # 0: compNone, 1: compGreeen, 2: compWhite, 3: compBG
            "mode": 0,
            "index": 0,
        },
    }

    green_screen_options_2 = {
        "num_effects": 2,
        "model_dir": model_dir,
        "GreenScreen": {
            "mode": 0,
            "comp_mode": 0,
            "index": 0,
        },
        "BackgroundBlur": {
            "index": 1,
        }
    }

    green_screen_options_3 = {
        "num_effects": 1,
        "model_dir": model_dir,
        "GreenScreen": {
            "mode": 0,
            "comp_mode": 3,
            "bg_file": "input1.jpg",
            "index": 0,
        },
    }

    pipeline_options = {
        "num_effects": 2,
        "model_dir": model_dir,
        "ArtifactReduction": {
            "mode": 1,
            "index": 0,
        },
        "SuperRes": {
            "resolution": 2160,
            "mode": 0,
            "index": 1,
        },
    }

    test_maxine_effect(artifact_reduction_options, "ArtifactReduction_output.mp4")

    test_maxine_effect(super_res_options, "SuperRes_output.mp4")
    
    test_maxine_effect(upscale_options, "Upscale_output.mp4")

    test_maxine_effect(artifact_reduction_upscale_options, "ArtiRedUpscale_output.mp4")

    test_maxine_effect(green_screen_options, "GreenScreen.mp4")

    test_maxine_effect(green_screen_options_2, "GreenScreen2.mp4")

    test_maxine_effect(green_screen_options_3, "GreenScreen3.mp4")

    test_maxine_effect(super_res_options, "SuperRes_output.mp4")

    test_maxine_effect(pipeline_options, "Pipeline_output.mp4")

    test_effect_perf(upscale_options, "1080p.flv", "Upscale_perf.mp4", 1)

if __name__ == "__main__":
    main()