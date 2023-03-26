import sys
import unittest

sys.path.append("../../..")
import bmf
from bmf import ServerGateway, module
from bmf import GraphMode, Module, Log, LogLevel, InputType, ProcessResult, Packet, Timestamp, scale_av_pts, \
    av_time_base, BmfCallBackType, VideoFrame, AudioFrame
import threading

sys.path.append("../")
from base_test.base_test_case import BaseTestCase


def process_thread(server_gateway, packet):
    result = server_gateway.process_work(packet)
    print("result is : " + str(result))


class TestServer(BaseTestCase):
    def test_single_video(self):
        input_video_path_1 = "../files/img.mp4"
        graph = bmf.graph({"dump_graph": 0})
        video_stream = graph.module('ffmpeg_decoder')
        video_stream['video'].pass_through().encode(
            video_stream['audio'],
            {
                "output_prefix": "./output_video_dir",
                "video_params": {
                    "codec": "h264",
                    "width": 640,
                    "height": 480,
                    "crf": "23",
                    "preset": "veryfast"
                }
            }
        ).output_stream()

        server_gateway = ServerGateway(graph)
        server_gateway.init()

        # create a packet and send it to the graph
        video_info_list1 = []
        video_info_list1.append({'input_path': input_video_path_1})
        data = {'type': InputType.VIDEO, 'input_path': video_info_list1}
        packet1 = Packet(data)
        packet1.timestamp = 1
        thread_ = threading.Thread(target=process_thread, args=(server_gateway, packet1))
        thread_.start()

        # create a packet and send it to the graph
        video_info_list2 = []
        video_info2 = {'input_path': input_video_path_1}
        video_info_list2.append(video_info2)
        data = {'type': InputType.VIDEO, 'input_path': video_info_list2}
        packet2 = Packet(data)
        packet2.timestamp = 2
        thread_ = threading.Thread(target=process_thread, args=(server_gateway, packet2))
        thread_.start()

        # finish the process
        server_gateway.close()

    def test_multiple_pictures(self):
        input_video_path_1 = "../files/blue.png"
        input_video_path_2 = '../files/overlay.png'

        graph = bmf.graph({"dump_graph": 1})

        video_stream = graph.input_stream('graph_input_name').module('ffmpeg_decoder')
        video_stream['video'].pass_through().encode(
            None,
            {
                "output_prefix": "./output_multi_pic_dir",
                "video_params": {
                    "codec": "jpg",
                    "width": 320,
                    "height": 240
                }
            }
        ).output_stream()

        server_gateway = ServerGateway(graph)
        server_gateway.init()

        pic_info_list = []
        # pic_dir = '../files/'
        for i in range(50):
            pic_info = {'input_path': input_video_path_1}
            pic_info_list.append(pic_info)
        for i in range(50):
            pic_info = {'input_path': input_video_path_2}
            pic_info_list.append(pic_info)
        data = {'type': InputType.PICTURELIST, 'input_path': pic_info_list}
        packet1 = Packet(data)
        packet1.timestamp = 1
        thread_ = threading.Thread(target=process_thread, args=(server_gateway, packet1))
        thread_.start()

        pic_info_list2 = []
        for i in range(50):
            pic_info = {'input_path': input_video_path_2}
            pic_info_list2.append(pic_info)
        for i in range(50):
            pic_info = {'input_path': input_video_path_1}
            pic_info_list2.append(pic_info)
        data = {'type': InputType.PICTURELIST, 'input_path': pic_info_list2}
        packet2 = Packet(data)
        packet2.timestamp = 2
        thread_ = threading.Thread(target=process_thread, args=(server_gateway, packet2))
        thread_.start()

        # finish the process
        server_gateway.close()

    def test_filter_condition(self):
        input_video_path = '../files/img.mp4'

        graph = bmf.graph({"dump_graph": 0})
        video_stream = graph.module('ffmpeg_decoder')
        video_stream['video'].vflip().pass_through().scale(100, 200).encode(
            video_stream['audio'],
            {
                "output_prefix": "./output_filter_condition_dir",
                "video_params": {
                    "codec": "h264",
                    "width": 640,
                    "height": 480,
                    "crf": "23",
                    "preset": "veryfast"
                }
            }
        ).output_stream()

        server_gateway = ServerGateway(graph)
        server_gateway.init()

        video_info_list = []
        video_info_list.append({'input_path': input_video_path})
        data = {'type': InputType.VIDEOLIST, 'input_path': video_info_list}
        packet1 = Packet(data)
        packet1.timestamp = 1
        thread_ = threading.Thread(target=process_thread, args=(server_gateway, packet1))
        thread_.start()

        video_info_list2 = []
        video_info_list2.append({'input_path': input_video_path})
        data2 = {'type': InputType.VIDEOLIST, 'input_path': video_info_list2}
        packet2 = Packet(data2)
        packet2.timestamp = 1
        thread_ = threading.Thread(target=process_thread, args=(server_gateway, packet2))
        thread_.start()

        # finish the process
        server_gateway.close()

    def test_first_module_is_not_decoder(self):
        input_file = "../../example/files/blue.png"

        graph = bmf.graph({"dump_graph": 1})
        complex_data = graph.module("complex_data")
        image_data = complex_data[0].decode()['video']
        detect_stream = module([image_data], 'pass_through', {"a": "b"})
        detect_stream.encode(
            None,
            {
                "format": "mjpeg",
                "video_params": {"codec": "jpg"},
                "output_prefix": "./first_module_is_not_decoder"  # ,"video_params": {"codec": "png"}
            }).output_stream()
        server_gateway = ServerGateway(graph)
        server_gateway.init()
        data = {"video_data": {'type': InputType.VIDEO, 'input_path': [{'input_path': input_file}]},
                "extra_data": "hello_world"}
        packet = Packet(data)
        packet.timestamp = 1
        thread_ = threading.Thread(target=process_thread, args=(server_gateway, packet))
        thread_.start()

        server_gateway.close()


    def test_all_results_at_one_time(self):
        input_video_path = "../files/img.mp4"

        graph = bmf.graph({"dump_graph": 1})

        video_stream = graph.module('ffmpeg_decoder')
        server_gateway = video_stream['video'].pass_through().vflip().encode(
            None,
            {
                "output_prefix": "./all_results_at_one_time",
                "video_params": {
                    "codec": "h264",
                    "width": 640,
                    "height": 480,
                    "crf": "23",
                    "preset": "veryfast"
                }
            }
        ).server(mode=1)

        for i in range(2):
            video_info_list1 = []
            video_info_list1.append({'input_path': input_video_path})
            data = {'type': InputType.VIDEO, 'input_path': video_info_list1}
            packet1 = Packet(data)
            packet1.timestamp = 1
            server_gateway.process_work(packet1)
        for i in range(2):
            video_info_list1 = []
            video_info_list1.append({'input_path': input_video_path})
            data = {'type': InputType.VIDEO, 'input_path': video_info_list1}
            packet1 = Packet(data)
            packet1.timestamp = 1
            server_gateway.process_work(packet1)

        res = server_gateway.request_for_res()
        server_gateway.close()
        assert len(res) == 4
        assert res['res_1'] == './all_results_at_one_time/1'
        assert res['res_2'] == './all_results_at_one_time/2'
        assert res['res_3'] == './all_results_at_one_time/3'
        assert res['res_4'] == './all_results_at_one_time/4'

    def test_specified_job_result_unblock(self):
        input_video_path = "../files/img.mp4"

        graph = bmf.graph({"dump_graph": 1})
        video_stream = graph.module('ffmpeg_decoder')
        server_gateway = video_stream['video'].pass_through().vflip().encode(
            None,
            {
                "output_prefix": "./specified_job_result_unblock",
                "video_params": {
                    "codec": "h264",
                    "width": 640,
                    "height": 480,
                    "crf": "23",
                    "preset": "veryfast"
                }
            }
        ).server(mode=1)

        video_info_list1 = []
        video_info_list1.append({'input_path': input_video_path})
        data = {'type': InputType.VIDEO, 'input_path': video_info_list1}
        packet1 = Packet(data)
        packet1.timestamp = 1
        server_gateway.process_work(packet1)

        video_info_list1 = []
        video_info_list1.append({'input_path': input_video_path})
        data = {'type': InputType.VIDEO, 'input_path': video_info_list1}
        packet1 = Packet(data)
        packet1.timestamp = 1
        server_gateway.process_work(packet1, "high_priority_job")

        video_info_list1 = []
        video_info_list1.append({'input_path': input_video_path})
        data = {'type': InputType.VIDEO, 'input_path': video_info_list1}
        packet1 = Packet(data)
        packet1.timestamp = 1
        server_gateway.process_work(packet1)

        res = server_gateway.get_by_job_name("high_priority_job")
        server_gateway.close()
        assert res is None

    def test_specified_job_result_block(self):
        input_video_path = "../files/img.mp4"

        graph = bmf.graph({"dump_graph": 1})
        video_stream = graph.module('ffmpeg_decoder')
        server_gateway = video_stream['video'].pass_through().vflip().encode(
            None,
            {
                "output_prefix": "./specified_job_result_block",
                "video_params": {
                    "codec": "h264",
                    "width": 640,
                    "height": 480,
                    "crf": "23",
                    "preset": "veryfast"
                }
            }
        ).server(mode=1)

        video_info_list1 = []
        video_info_list1.append({'input_path': input_video_path})
        data = {'type': InputType.VIDEO, 'input_path': video_info_list1}
        packet1 = Packet(data)
        packet1.timestamp = 1
        server_gateway.process_work(packet1)

        video_info_list1 = []
        video_info_list1.append({'input_path': input_video_path})
        data = {'type': InputType.VIDEO, 'input_path': video_info_list1}
        packet1 = Packet(data)
        packet1.timestamp = 1
        server_gateway.process_work(packet1, "high_priority_job")

        video_info_list1 = []
        video_info_list1.append({'input_path': input_video_path})
        data = {'type': InputType.VIDEO, 'input_path': video_info_list1}
        packet1 = Packet(data)
        packet1.timestamp = 1
        server_gateway.process_work(packet1)

        res = server_gateway.get_by_job_name("high_priority_job", block=True)
        server_gateway.close()
        assert len(res) == 1

    def test_pop_from_result_queue(self):
        input_video_path = "../files/img.mp4"

        graph = bmf.graph({"dump_graph": 1})
        video_stream = graph.module('ffmpeg_decoder')
        server_gateway = video_stream['video'].pass_through().vflip().encode(
            None,
            {
                "output_prefix": "./pop_from_result_queue",
                "video_params": {
                    "codec": "h264",
                    "width": 640,
                    "height": 480,
                    "crf": "23",
                    "preset": "veryfast"
                }
            }
        ).server(mode=1)

        video_info_list1 = []
        video_info_list1.append({'input_path': input_video_path})
        data = {'type': InputType.VIDEO, 'input_path': video_info_list1}
        packet1 = Packet(data)
        packet1.timestamp = 1
        server_gateway.process_work(packet1)

        video_info_list1 = []
        video_info_list1.append({'input_path': input_video_path})
        data = {'type': InputType.VIDEO, 'input_path': video_info_list1}
        packet1 = Packet(data)
        packet1.timestamp = 1
        server_gateway.process_work(packet1, "high_priority_job")

        res_list = []
        while not server_gateway.empty_result():
            res = server_gateway.get_front_result()
            res_list.append(res)
        server_gateway.close()

        assert str(res_list[0])=="{'res_1': './pop_from_result_queue/1'}"
        assert str(res_list[1]) == "{'high_priority_job': './pop_from_result_queue/2'}"


if __name__ == '__main__':
    unittest.main()
