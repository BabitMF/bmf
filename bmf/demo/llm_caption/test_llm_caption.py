import bmf
from llm_caption import llm_caption

def main():
    input_path = "../LLM_video_preprocessing/big_bunny_10s_30fps.mp4"
    graph = bmf.graph({"dump_graph": 0})
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
                                                   "batch_size": 1,
                                                   "multithreading": False,
                                                   "model": "Qwen2_VL"
                                                   # "max_threads": 2,
                                          }).run()

if __name__ == "__main__":
    main()
