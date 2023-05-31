import sys
import time
import unittest
import numpy as np
import bmf.hml.hmp as mp

sys.path.append("../../..")
import bmf
from bmf import GraphMode, Module, Log, LogLevel, InputType, ProcessResult, Packet, Timestamp, scale_av_pts, \
    av_time_base, BmfCallBackType, VideoFrame, AudioFrame, BMFAVPacket
import threading
import timeout_decorator

sys.path.append("../")
from base_test.base_test_case import BaseTestCase


def push_file(file_name, graph, video_stream1, video_stream2, pts):
    f = open(file_name, "rb")
    while(1):
        lines = f.read(1000)
        if len(lines) == 0:
            break
        pkt = BMFAVPacket(len(lines))
        memview = pkt.data.numpy()
        memview[:] = np.frombuffer(lines, dtype='uint8')
        pkt.pts = pts
        pts += 1
        packet = Packet(pkt)
        packet.timestamp = pts
        graph.fill_packet(video_stream1.get_name(), packet, True)
        graph.fill_packet(video_stream2.get_name(), packet, True)
    f.close()
    return pts


class TestPushData(BaseTestCase):
    @timeout_decorator.timeout(seconds=120)
    def test_push_pkt_into_decoder(self):
        output_path = "./aac.mp4"

        self.remove_result_data(output_path)
        
        graph = bmf.graph({"dump_graph": 1})

        video_stream = graph.input_stream("outside_raw_video")
        decode_stream = video_stream.decode()
        bmf.encode(None,decode_stream["audio"],{"output_path": output_path})
        
        graph.run_wo_block(mode = GraphMode.PUSHDATA)
        pts_ = 0
        for index in range(100,105):
            file_name = "../files/aac_slice/"+str(index)+".aac"
            with open(file_name, "rb") as fp:
                lines = fp.read()    
                buf = BMFAVPacket(len(lines))
                buf.data.numpy()[:] = np.frombuffer(lines, dtype=np.uint8)
                buf.pts = pts_

                packet = Packet(buf)
                pts_ += 1
                packet.timestamp = pts_
                start_time = time.time()
                graph.fill_packet(video_stream.get_name(), packet, True)
        graph.fill_packet(video_stream.get_name(),Packet.generate_eof_packet())
        graph.close()
    
    @timeout_decorator.timeout(seconds=120)
    def test_push_video_frame_into_encode(self):

        output_path = "./push_data_output.mp4"

        self.remove_result_data(output_path)

        graph = bmf.graph({"dump_graph": 1})

        video_stream = graph.input_stream("outside_raw_video")
        bmf.encode(
            video_stream,
            None,
            {
                "output_path": output_path,
                "video_params": {
                    "codec": "h264",
                    "width": 640,
                    "height": 480,
                    "crf": "23",
                    "preset": "veryfast",
                    "vsync": "vfr"
                }
            }
        )
        graph.run_wo_block(mode = GraphMode.PUSHDATA)

        count = 50
        pts_ = 0
        while count > 0:
            frame = VideoFrame(640, 480, mp.PixelInfo(mp.kPF_YUV420P))
            frame.pts = pts_
            pts_ += 1
            packet = Packet(frame)
            count -= 1
            graph.fill_packet(video_stream.get_name(), packet)
            print("push data into inputstream")

        graph.fill_packet(video_stream.get_name(), Packet.generate_eof_packet())
        graph.close()


    @timeout_decorator.timeout(seconds=120)
    def test_push_raw_stream_into_decoder(self):
        input_video_content = "../files/video_content.txt"
        input_content_size = "../files/video_length.txt"
        output_path = "./push_pkt_output.mp4"
        expect_result = '../transcode/push_pkt_output.mp4|480|640|7.60|MOV,MP4,M4A,3GP,3G2,MJ2|911360|851968|h264|{"fps": "27.63"}'

        self.remove_result_data(output_path)

        graph = bmf.graph({"dump_graph": 1})

        video_stream = graph.input_stream("outside_raw_video")
        
        decode_stream = video_stream.module(
            'ffmpeg_decoder', 
            option={
                "video_codec": "h264", 'video_time_base': "1,30000",
                "push_raw_stream": 1
            }
        )

        encode_stream = decode_stream['video'].encode(
            None,
            {
                "output_path": output_path,
                "video_params": {
                    "codec": "h264",
                    "width": 640,
                    "height": 480,
                    "max_fr": 30,
                    "crf": "23",
                    "preset": "veryfast"
                }
            }
        )
        graph.run_wo_block(mode = GraphMode.PUSHDATA)

        f_cont = open(input_video_content,'rb')
        f_size = open(input_content_size,'r')

        pts_ = 0
        timestamp = 0
        lines = f_size.readlines()
        for size in lines:
            pkt = BMFAVPacket(int(size))
            memview = pkt.data.numpy()
            memview[:] = np.frombuffer(f_cont.read(int(size)), dtype='uint8')
            pkt.pts = pts_
            packet = Packet(pkt)
            packet.timestamp = timestamp
            pts_ += 1001
            timestamp += 1
            graph.fill_packet(video_stream.get_name(), packet)

        graph.fill_packet(video_stream.get_name(), Packet.generate_eof_packet())
        graph.close()
        f_size.close()
        f_cont.close()
        self.check_video_diff(output_path, expect_result)

    @timeout_decorator.timeout(seconds=120)
    def test_push_pkt_into_decoder_multi_output(self):
        input_file1 = "../files/lark_stream0.flv"
        output_path = "./stream_copy.mp4"
        expect_result = 'stream_copy.mp4|1080|1920|5.920000|MOV,MP4,M4A,3GP,3G2,MJ2|2834318|2097396|h264|{"fps": "29.998265996185193"}'
        output_path1 = "./output1.mp4"
        expect_result1 = 'output1.mp4|1080|1920|5.944000|MOV,MP4,M4A,3GP,3G2,MJ2|980537|728539|h264|{"fps": "30.0"}'
        output_path2 = "./output2.mp4"
        expect_result2 = 'output2.mp4|720|1080|5.944000|MOV,MP4,M4A,3GP,3G2,MJ2|427518|317646|h264|{"fps": "30.0"}'
        output_path3 = "./output3.mp4"
        expect_result3 = 'output3.mp4|360|540|5.944000|MOV,MP4,M4A,3GP,3G2,MJ2|222822|165557|h264|{"fps": "30.0"}'
        self.remove_result_data(output_path)
        self.remove_result_data(output_path1)
        self.remove_result_data(output_path2)
        self.remove_result_data(output_path3)

        graph = bmf.graph({"dump_graph": 1})

        video_stream1 = graph.input_stream("outside_raw_video")
        video_stream2 = graph.input_stream("outside_raw_video2")
        decode_stream = video_stream1.decode()
        decode_stream2 = video_stream2.decode({'video_codec': "copy", 'audio_codec': "copy"})
        video = decode_stream["video"]
        video_params = {
            "codec": "h264",
            "preset": "medium",
            "maxrate": "2000000.0",
            "bufsize": "4000000.0",
            "crf": "23",
            "x264-params": "ssim=1",
            "vsync": "vfr",
            "r": "30.0",
            "pix_fmt": "yuv420p",
            "color_range": "tv"
        },
        audio_params = {
            "codec": "libfdk_aac",
            "bit_rate": 64000,
            "sample_rate": 44100,
            "channels": 2
        },
        mux_params = {
            "fflags": "igndts",
            "movflags": "+faststart+use_metadata_tags",
            "max_interleave_delta": "0",
            "max_muxing_queue_size": "9999",
            "map_metadata": "-1",
            "map_chapters": "-1"
        }
        video.scale(1920, 1080).encode(decode_stream["audio"], {"output_path": output_path1, "video_params": video_params, "audio_params": audio_params, "mux_params": mux_params})
        video.scale(1080, 720).encode(decode_stream["audio"], {"output_path": output_path2, "video_params": video_params, "audio_params": audio_params, "mux_params": mux_params})
        video.scale(540, 360).encode(decode_stream["audio"], {"output_path": output_path3, "video_params": video_params, "audio_params": audio_params, "mux_params": mux_params})

        bmf.encode(decode_stream2["video"], decode_stream2["audio"], {"output_path": output_path})

        graph.run_wo_block(mode=GraphMode.PUSHDATA)
        pts = 0
        push_file(input_file1, graph, video_stream1, video_stream2, pts)
        print("send eof")
        graph.fill_packet(video_stream1.get_name(), Packet.generate_eof_packet(), True)
        graph.fill_packet(video_stream2.get_name(), Packet.generate_eof_packet(), True)
        graph.close()
        self.check_video_diff(output_path, expect_result)
        self.check_video_diff(output_path1, expect_result1)
        self.check_video_diff(output_path2, expect_result2)
        self.check_video_diff(output_path3, expect_result3)


    @timeout_decorator.timeout(seconds=120)
    def test_cut_off_stream_graph(self):
        input_file1 = "../files/lark_stream0.flv"
        input_file2 = "../files/lark_stream1.flv"
        output_path = "./stream_copy.mp4"
        expect_result = 'stream_copy.mp4|1080|1920|29.493000|MOV,MP4,M4A,3GP,3G2,MJ2|4158975|15332582|h264|{"fps": "29.99488141955298"}'
        output_path1 = "./output1.mp4"
        expect_result1 = 'output1.mp4|1080|1920|29.567000|MOV,MP4,M4A,3GP,3G2,MJ2|3713941|13726264|h264|{"fps": "30.0"}'
        output_path2 = "./output2.mp4"
        expect_result2 = 'output2.mp4|720|1080|29.567000|MOV,MP4,M4A,3GP,3G2,MJ2|2244088|8293869|h264|{"fps": "30.0"}'
        output_path3 = "./output3.mp4"
        expect_result3 = 'output3.mp4|360|540|29.567000|MOV,MP4,M4A,3GP,3G2,MJ2|690303|2551274|h264|{"fps": "30.0"}'
 
        self.remove_result_data(output_path)
        self.remove_result_data(output_path1)
        self.remove_result_data(output_path2)
        self.remove_result_data(output_path3)
        graph = bmf.graph({"dump_graph": 1})

        video_stream1 = graph.input_stream("outside_raw_video")
        video_stream2 = graph.input_stream("outside_raw_video2")
        decode_stream = video_stream1.decode()
        decode_stream2 = video_stream2.decode({'video_codec': "copy", 'audio_codec': "copy"})
        video = decode_stream["video"]
        video_params = {
            "codec": "h264",
            "preset": "medium",
            "maxrate": "2000000.0",
            "bufsize": "4000000.0",
            "crf": "23",
            "x264-params": "ssim=1",
            "vsync": "vfr",
            "r": "30.0",
            "pix_fmt": "yuv420p",
            "color_range": "tv"
        },
        audio_params = {
            "codec": "libfdk_aac",
            "bit_rate": 64000,
            "sample_rate": 44100,
            "channels": 2
        },
        mux_params = {
            "fflags": "igndts",
            "movflags": "+faststart+use_metadata_tags",
            "max_interleave_delta": "0",
            "max_muxing_queue_size": "9999",
            "map_metadata": "-1",
            "map_chapters": "-1"
        }
        video.scale(1920, 1080).encode(decode_stream["audio"], {"output_path": output_path1, "video_params": video_params, "audio_params": audio_params, "mux_params": mux_params})
        video.scale(1080, 720).encode(decode_stream["audio"], {"output_path": output_path2, "video_params": video_params, "audio_params": audio_params, "mux_params": mux_params})
        video.scale(540, 360).encode(decode_stream["audio"], {"output_path": output_path3, "video_params": video_params, "audio_params": audio_params, "mux_params": mux_params})

        bmf.encode(decode_stream2["video"], decode_stream2["audio"], {"output_path": output_path})
        graph.run_wo_block(mode=GraphMode.PUSHDATA)
        pts = 0
        pts = push_file(input_file1, graph, video_stream1, video_stream2, pts)
        time.sleep(1)
        print("push discontinue file")
        pkt = BMFAVPacket()
        pts += 1
        pkt.pts = pts
        assert(pkt.nbytes == 0)
        packet = Packet(pkt)
        packet.timestamp = pts
        graph.fill_packet(video_stream1.get_name(), packet)
        graph.fill_packet(video_stream2.get_name(), packet)
        print("push another file")
        push_file(input_file2, graph, video_stream1, video_stream2, pts)
        graph.fill_packet(video_stream1.get_name(), Packet.generate_eof_packet(), True)
        graph.fill_packet(video_stream2.get_name(), Packet.generate_eof_packet(), True)
        graph.close()
        self.check_video_diff(output_path, expect_result)
        self.check_video_diff(output_path1, expect_result1)
        self.check_video_diff(output_path2, expect_result2)
        self.check_video_diff(output_path3, expect_result3)

    @timeout_decorator.timeout(seconds=120)
    def test_push_raw_audio_into_decoder(self):
        input_audio_content = "../files/audio_content.txt"
        input_content_size = "../files/audio_length.txt"
        output_path = "./push_audio_output.mp4"
        expect_result = './push_audio_output.mp4|0|0|7.617000|MOV,MP4,M4A,3GP,3G2,MJ2|136092|129577||{}'
        self.remove_result_data(output_path)

        graph = bmf.graph({"dump_graph": 1})

        audio_stream = graph.input_stream("outside_raw_audio")
        
        decode_stream = audio_stream.module(
            'ffmpeg_decoder', 
            option={
                "audio_codec": "aac",
                "channels": 2,
                "sample_rate": 44100,
                "push_raw_stream": 1
            }
        )

        encode_stream = bmf.encode(
            None,
            decode_stream['audio'],
            {
                "output_path": output_path
            }
        )
        graph.run_wo_block(mode = GraphMode.PUSHDATA)

        f_cont = open(input_audio_content,'rb')
        f_size = open(input_content_size,'r')

        pts_ = 0
        lines = f_size.readlines()
        for size in lines:
            pkt = BMFAVPacket(int(size))
            memview = pkt.data.numpy()
            memview[:] = np.frombuffer(f_cont.read(int(size)), dtype=memview.dtype)
            pkt.pts = pts_
            packet = Packet(pkt)
            packet.timestamp = pts_
            pts_ += 20
            graph.fill_packet(audio_stream.get_name(), packet)

        graph.fill_packet(audio_stream.get_name(), Packet.generate_eof_packet())
        graph.close()
        f_size.close()
        f_cont.close()

        self.check_video_diff(output_path, expect_result)

if __name__ == '__main__':
    unittest.main()
