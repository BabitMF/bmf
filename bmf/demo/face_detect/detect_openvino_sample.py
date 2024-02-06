import sys

sys.path.append("../../..")
import bmf
from bmf import Log, LogLevel
import os

def main():
    graph = bmf.graph({"dump_graph": 1})

    openvino_face_detect = bmf.create_module('openvino_face_detect', {
        "model_path": "../../models/version-RFB-640.onnx",
        "label_to_frame": 1,
        "threads": 2
    })
    video_stream = graph.decode({'input_path': "../../files/face.mp4"})
    #video_stream = graph.download({
    #    'input_url': 'https://github.com/fromwhzz/test_video/raw/master/face.mp4',
    #    'local_path': '../../files/face_test.mp4'
    #}).decode()
    detect_stream = video_stream['video'].module('openvino_face_detect',
                                                 pre_module=openvino_face_detect)
    detect_stream[0].encode(None, {"output_path": "../../files/out.mp4"}).run()


if __name__ == '__main__':
    pid = os.getpid()
    core_list = [0, 1]#if use multi openvino process, it's best to bind the core to avoid preemption of CPU resources. 
    os.sched_setaffinity(pid, core_list)
    Log.set_log_level(LogLevel.ERROR)
    main()
