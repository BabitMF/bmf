import sys
sys.path.append("../../")
import bmf
import time

def json_param_transfer():
    graph = bmf.graph({'dump_graph': 1})
    video = graph.decode({
        "input_path": "../../files/big_bunny_10s_30fps.mp4",
        "video_codec": "copy"
    })["video"]
    fps = 1
    module_path = './libcpp_module.so'
    module_entry = 'cpp_module:cpp_module'
    # the graph is triggered by a decoder and then connect cpp module -> python module -> cpp module
    # to show the json can be transferred and parsed between cpp and python module easily
    cpp_json = video.c_module("cpp_module",
                                      module_path,
                                      module_entry,
                                      )
    python_paser = cpp_json.module('python_module')
    cpp_paser = python_paser.c_module("cpp_module",
                                      module_path,
                                      module_entry,
                                      ).run()


if __name__ == "__main__":
    start_time = time.time()
    json_param_transfer()
    end_time = time.time()
    print("process time:", end_time - start_time)
