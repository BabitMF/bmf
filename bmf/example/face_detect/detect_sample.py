import sys

sys.path.append("../../..")
import bmf
from bmf import Log,LogLevel


def main():
    graph = bmf.graph({"dump_graph": 1})

    onnx_face_detect = bmf.create_module('onnx_face_detect', {
        "model_path": "version-RFB-640.onnx",
        "label_to_frame": 1
    })
    video_stream = graph.decode({'input_path': "../files/face_test_small.mp4"})
    #video_stream = graph.download({
    #    'input_url': 'https://github.com/fromwhzz/test_video/raw/master/face.mp4',
    #    'local_path': '../files/face_test.mp4'
    #}).decode()
    detect_stream = video_stream['video'].module('onnx_face_detect', pre_module=onnx_face_detect)
    detect_stream[0].encode(
        None, {"output_path": "../files/out.mp4"}
    )
    detect_stream[1].module("upload").run()


if __name__ == '__main__':
    Log.set_log_level(LogLevel.ERROR)
    main()
