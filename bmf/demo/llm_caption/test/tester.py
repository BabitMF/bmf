import os
import sys
import bmf

def main(args):
    input_path = args[1]
    model_name = args[2]
    batch_size = int(args[3])
    output_path = args[4]

    parent_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
    sys.path.append(parent_dir) 

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
    video['video'].module('llm_caption', {"result_path": output_path,
                                                   "batch_size": batch_size,
                                                   "multithreading": False,
                                                   "model": model_name
                                                   # "max_threads": 2,
                                          }).run()

if __name__ == "__main__":
    main(sys.argv)


