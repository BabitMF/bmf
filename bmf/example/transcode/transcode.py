import sys
import time
import unittest

sys.path.append("../../../3rd_party/pyav")
sys.path.append("../../../")
import bmf


def test_simple():
    input_video_path = "../files/img.mp4"
    output_path = "./simple.mp4"
    expect_result = '../transcode/simple.mp4|240|320|7.617000|MOV,MP4,M4A,3GP,3G2,MJ2|369643|351947|h264|' \
                    '{"fps": "30.0662251656"}'

    # create graph
    graph = bmf.graph()

    # decode
    video = graph.decode({
        "input_path": input_video_path
    })

    (
        bmf.encode(
            video['video'],
            video['audio'],
            {
                "output_path": output_path,
                "video_params": {
                    "codec": "h264",
                    "width": 320,
                    "height": 240,
                    "crf": 23,
                    "preset": "veryfast"
                },
                "audio_params": {
                    "codec": "aac",
                    "bit_rate": 128000,
                    "sample_rate": 44100,
                    "channels": 2
                }
            }
        )
            .run()
    )


def test_audio():
    input_video_path = "../files/img.mp4"
    output_path = "./audio.mp4"
    expect_result = '../transcode/audio.mp4|0|0|7.617000|MOV,MP4,M4A,3GP,3G2,MJ2|136092|129577||{}'
    # create graph
    graph = bmf.graph()

    # decode
    video = graph.decode({
        "input_path": input_video_path
    })

    (
        bmf.encode(
            None,
            video['audio'],
            {
                "output_path": output_path,
                "audio_params": {
                    "codec": "aac",
                    "bit_rate": 128000,
                    "sample_rate": 44100,
                    "channels": 2
                }
            }
        )
            .run()
    )


def test_video():
    input_video_path_1 = "../files/header.mp4"
    input_video_path_2 = "../files/header.mp4"
    input_video_path_3 = '../files/img.mp4'
    logo_video_path_1 = "../files/xigua_prefix_logo_x.mov"
    logo_video_path_2 = "../files/xigua_loop_logo2_x.mov"
    output_path = "./video.mp4"
    expect_result = '../transcode/video.mp4|720|1280|13.652000|MOV,MP4,M4A,3GP,3G2,MJ2|3902352|6659364|h264|' \
                    '{"fps": "69.7605893186"}'
    # some parameters
    output_width = 1280
    output_height = 720
    logo_width = 320
    logo_height = 144

    # create graph
    my_graph = bmf.graph()

    # tail video
    tail = my_graph.decode({'input_path': input_video_path_1})

    # header video
    header = my_graph.decode({'input_path': input_video_path_2})

    # main video
    video = my_graph.decode({'input_path': input_video_path_3})

    # logo video
    logo_1 = (
        my_graph.decode({'input_path': logo_video_path_1})['video']
            .scale(logo_width, logo_height)
    )
    logo_2 = (
        my_graph.decode({'input_path': logo_video_path_2})['video']
            .scale(logo_width, logo_height)
            .ff_filter('loop', loop=-1, size=991)
            .ff_filter('setpts', 'PTS+3.900/TB')
    )

    # main video processing
    main_video = (
        video['video'].scale(output_width, output_height)
            .overlay(logo_1, repeatlast=0)
            .overlay(logo_2,
                     x='if(gte(t,3.900),960,NAN)',
                     y=0,
                     shortest=1)
    )

    # concat video
    concat_video = (
        bmf.concat(header['video'].scale(output_width, output_height),
                   main_video,
                   tail['video'].scale(output_width, output_height),
                   n=3)
    )

    # concat audio
    concat_audio = (
        bmf.concat(header['audio'],
                   video['audio'],
                   tail['audio'],
                   n=3, v=0, a=1)
    )

    (
        bmf.encode(concat_video,
                   concat_audio,
                   {
                       "output_path": output_path,
                       "video_params": {
                           "codec": "h264",
                           "width": 1280,
                           "height": 720,
                           "preset": "veryfast",
                           "crf": "23",
                           "x264-params": "ssim=1:psnr=1"
                       },
                       "audio_params": {
                           "codec": "aac",
                           "bit_rate": 128000,
                           "sample_rate": 48000,
                           "channels": 2
                       },
                       "mux_params": {
                           "fflags": "+igndts",
                           "movflags": "+faststart+use_metadata_tags",
                           "max_interleave_delta": "0"
                       }
                   })
            .run()
    )


def test_cb():
    input_video_path = "../files/img.mp4"
    output_path = "./cb.mp4"
    expect_result = '../transcode/cb.mp4|240|320|7.615000|MOV,MP4,M4A,3GP,3G2,MJ2|366635|348991|h264|' \
                    '{"fps": "30.0662251656"}'
    # create graph
    graph = bmf.graph()

    def cb(para):
        print(para)

    graph.add_user_callback(bmf.BmfCallBackType.LATEST_TIMESTAMP, cb)

    # decode
    video = graph.decode({
        "input_path": input_video_path
    })

    (
        bmf.encode(
            video['video'],
            video['audio'],
            {
                "output_path": output_path,
                "video_params": {
                    "codec": "h264",
                    "width": 320,
                    "height": 240,
                    "crf": "23",
                    "preset": "veryfast"
                }
            }
        )
            .run()
    )


def compareProfile(graph_file):
    import os
    import time
    ffmpeg_comand = "python3 compare.py " + graph_file + " ffmpeg"
    start_time = time.time()
    os.system(ffmpeg_comand)
    end_time = time.time()
    runtime = end_time - start_time
    print("ffmpeg_comand :", runtime)

    start_time = time.time()
    python_command = "python3 compare.py " + graph_file + " pythonEngine"
    os.system(python_command)
    end_time = time.time()
    runtime = end_time - start_time
    print("python_command :", runtime)

    start_time = time.time()
    c_command = "python3 compare.py " + graph_file + " cEngine"
    os.system(c_command)
    end_time = time.time()
    runtime = end_time - start_time
    print("c_command :", runtime)


def compare():
    input_video_path = "../files/img.mp4"
    output_path = "./simple.mp4"

    # create graph
    graph = bmf.graph()

    # decode
    video = graph.decode({
        "input_path": input_video_path
    })
    graph_file = "graph.json"
    (
        bmf.encode(
            video['video'],
            video['audio'],
            {
                "output_path": output_path,
                "video_params": {
                    "codec": "h264",
                    "width": 320,
                    "height": 240,
                    "crf": 23,
                    "preset": "veryfast"
                },
                "audio_params": {
                    "codec": "aac",
                    "bit_rate": 128000,
                    "sample_rate": 44100,
                    "channels": 2
                }
            }
        )
            .generateConfig(graph_file)
    )
    compareProfile(graph_file)


if __name__ == '__main__':
    # test_simple()
    # test_audio()
    # test_video()
    # test_cb()
    compare()
