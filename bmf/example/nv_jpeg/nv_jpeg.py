import bmf
import time
from bmf.builder.ff_filter import decode

def extract_frame_test():
    graph = bmf.graph({'dump_graph':1})
    video = graph.decode({
        "input_path": "/mnt/lark.mp4",
        "video_params": {
            "hwaccel": "cuda",
            "hwaccel_output_format": "cuda",
        }
        })["video"]
    fps = 1
    #fps_video = video.fps(fps).c_module("c_ffmpeg_filter", option={'name':'scale_npp', 'para':'w=640:h=360'})
    #fps_video = video.c_module("c_ffmpeg_filter", option={'name':'scale_npp', 'para':'w=640:h=360'})
    module_path = './libjpeg_encoder.so'
    module_entry = 'jpeg_encoder:jpeg_encoder'
    encoder = video.fps(fps).c_module("jpeg_encoder", module_path, module_entry, option={"width": 640, "height": 360})
    encoder.module("file_io",option={"save_dir":"./"}).run()
    
if __name__ == "__main__":
    start_time = time.time()
    extract_frame_test()
    end_time = time.time()
    print("process time:",end_time-start_time)
