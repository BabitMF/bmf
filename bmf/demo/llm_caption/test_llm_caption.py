import bmf
from llm_caption import llm_caption
import sys
import torch

def main(args):
    input_path = "../LLM_video_preprocessing/big_bunny_10s_30fps.mp4"
    graph = bmf.graph({"dump_graph": 0})

    torch.multiprocessing.set_start_method("spawn")

    model = "Qwen_2_5_VL_3b"
    if len(args) == 2:
        model = args[1]
    video = graph.decode({
        "input_path": input_path,
        "video_params": {
            "hwaccel": "cuda",
            "extract_frames": {
                "fps": 1
            }
        }
    })
    video['video'].module('llm_caption', {"result_path": "result.json",
                                                   "batch_size": 5,
                                                   "multithreading": False,
                                                   "model": model
                                                   # "max_threads": 2,
                                          }).run()

if __name__ == "__main__":
    main(sys.argv)
