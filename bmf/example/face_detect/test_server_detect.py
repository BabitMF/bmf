import sys
import time
sys.path.append("../../..")
import bmf
from bmf import Module, Log, LogLevel, InputType, ProcessResult, Packet, Timestamp, scale_av_pts, av_time_base, \
    BmfCallBackType, VideoFrame, AudioFrame
from bmf import ServerGateway
import threading



def process_thread(server_gateway, packet):
    result = server_gateway.process_work(packet)
    print("result is : "+str(result))


def single_video():
    graph = bmf.graph({"dump_graph": 1})

    video_stream = graph.module('ffmpeg_decoder')
    detect_stream=video_stream['video'].module('onnx_face_detect', {
        "model_path": "version-RFB-640.onnx",
        "label_to_frame": 1
    })
    detect_stream[0].encode(
        None,
        {
            "output_prefix": "../files/output_video_dir",
            "video_params": {
                "codec": "h264",
                "width": 640,
                "height": 480,
                "crf": "23",
                "preset": "veryfast"
            }
        }
    ).output_stream()
    detect_stream[1].module("upload").output_stream()
    server_gateway = ServerGateway(graph)
    server_gateway.init()

    # create a packet and send it to the graph
    packet1 = Packet()
    packet1.set_timestamp(1)
    video_info_list1 = []
    video_info_list1.append({'input_path': '../files/img.mp4'})
    data = {'type': InputType.VIDEO, 'input_path': video_info_list1}
    packet1.set_data(data)
    thread_ = threading.Thread(target=process_thread, args=(server_gateway, packet1))
    thread_.start()

    # time.sleep(100)
    # create a packet and send it to the graph
    packet2 = Packet()
    packet2.set_timestamp(2)
    video_info_list2 = []
    video_info2 = {'input_path': '../files/header.mp4'}
    video_info_list2.append(video_info2)
    data = {'type': InputType.VIDEO, 'input_path': video_info_list2}
    packet2.set_data(data)
    thread_ = threading.Thread(target=process_thread, args=(server_gateway, packet2))
    thread_.start()

    # finish the process
    server_gateway.close()

if __name__ == '__main__':
    start_time = time.time()

    Log.set_log_level(LogLevel.DEBUG)

    single_video()
    # multiple_pictures()
    # multiple_videos()
    # filter_condition()

    print("Total time:", (time.time() - start_time))
