import sys
import time

import bmf

def one_to_n_transcode():
    output_path1 = "./video1.mp4"
    output_path2 = "./video2.mp4"
    output_path3 = "./video3.mp4"
    output_path4 = "./video4.mp4"
    input_video_path = '../../files/big_bunny_1min_30fps.mp4'

    my_graph = bmf.graph({'dump_graph': 1})
    decode_stream = my_graph.decode({'input_path': input_video_path})

    stream1 = bmf.module([decode_stream['video']], "c_ffmpeg_filter", option={
                'name': 'scale',
                'para': 'w=720:h=480'},
                scheduler = 2)
    stream2 = bmf.module([decode_stream['video']], "c_ffmpeg_filter", option={
                'name': 'scale',
                'para': 'w=720:h=480'},
                scheduler = 3)
    stream3 = bmf.module([decode_stream['video']], "c_ffmpeg_filter", option={
                'name': 'scale',
                'para': 'w=720:h=480'},
                scheduler = 4)
    stream4 = bmf.module([decode_stream['video']], "c_ffmpeg_filter", option={
                'name': 'scale',
                'para': 'w=720:h=480'},
                scheduler = 5)
    enc_param = {
                    "codec": "h264",
                    "preset": "veryfast",
                    "crf": "23",
                    "vsync": "vfr",
                    "max_fr": 60
                }

    enc_stream1 = bmf.module([stream1], "c_ffmpeg_encoder", option={
            "output_path": output_path1,
            "video_params": enc_param},
            scheduler = 6)
    enc_stream2 = bmf.module([stream2], "c_ffmpeg_encoder", option={
            "output_path": output_path2,
            "video_params": enc_param},
            scheduler = 7)
    enc_stream3 = bmf.module([stream3], "c_ffmpeg_encoder", option={
            "output_path": output_path3,
            "video_params": enc_param},
            scheduler = 8)
    enc_stream3 = bmf.module([stream4], "c_ffmpeg_encoder", option={
            "output_path": output_path4,
            "video_params": enc_param},
            scheduler = 9)

    my_graph.set_option({'optimize_graph': False})
    my_graph.run()

if __name__ == '__main__':
    t1 = time.time()
    one_to_n_transcode()
    tlast = (time.time() - t1) * 1000
    print("BMF time cost (ms):", tlast)
