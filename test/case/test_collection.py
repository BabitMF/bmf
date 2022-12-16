import sys
sys.path.append("../../")
import bmf
import time


def test_3_concat():
    input_video_path = "../files/1min.mp4"
    output_path = "./3_concat.mp4"

    # create graph
    my_graph = bmf.graph({
        "dump_graph": 1,
        "graph_name": "3_concat"
    })

    # main video
    video_1 = my_graph.decode({'input_path': input_video_path})

    video_2 = my_graph.decode({'input_path': input_video_path})

    video_3 = my_graph.decode({'input_path': input_video_path})

    video1 = video_1['video'].vflip().pass_through().scale(100,200)
    video2 = video_2['video'].vflip().pass_through().scale(100,200)
    video3 = video_3['video'].vflip().pass_through().scale(100,200)

    # video1_a = video_1['audio']
    # video2_a = video_2['audio']
    # video3_a = video_3['audio']

    concat_video = (
        bmf.concat(video1,video2,video3,n=3,v=1,a=0)
    )

    # concat_audio = (
    #     bmf.concat(video1_a,video2_a,video3_a,n=3,v=0,a=1)
    # )

    # encode
    (
        bmf.encode(concat_video, None, {
            "output_path": output_path,
            "video_params": {
                "width": 1280,
                "height": 720
            }
        }).run()
    )


def test_triangle_concat():
    input_video_path = "../files/img.mp4"
    output_path = "./triangle_concat.mp4"

    # create graph
    my_graph = bmf.graph({
        "dump_graph": 1,
        "graph_name": "triangle_concat"
    })

    video1 = my_graph.decode({'input_path': input_video_path})['video']
    video2 = my_graph.decode({'input_path': input_video_path})['video']

    v_l = video1.split()
    v1=v_l[0]
    v2=v_l[1]
    v1 = v1.loop(loop=-1,start=0,size=1).trim(duration=0.5)
    v3 = bmf.concat(v1,v2, n=2, v=1, a=0)
    v4 = bmf.concat(v3, video2, n=2, v=1, a=0)
    (
        bmf.encode(v4, None, {
            "output_path": output_path,
            "video_params": {
                "width": 1280,
                "height": 720
            }
        }).run()
    )


def test_last_pic():
    input_video_path_ = '../files/img.mp4'
    output_path = "./last_pic.mp4"
    my_graph = bmf.graph({"dump_graph": 1,"graph_name":"last_pic"})
    # main video
    video_decoder = my_graph.decode({'input_path': input_video_path_})
    video = video_decoder['video']
    v_list = video.split()
    v1 = v_list[0]
    v2 = v_list[1]
    v1 = v1.module("get_last_pic_from_video",option={'num':1}).loop(loop=-1,size=2,start=0).trim(start=0, duration=30)
    v1 = v1.pass_through()
    # v2 = v2.pass_through()
    v = bmf.concat(v2, v1, n=2, v=1, a=0)
    # v = bmf.module([v1,v2],"bmf_concat",{'num':2,'type':'video'})
    (
        v.encode(None, {
            "output_path": output_path,
            "video_params": {
                "width": 640,
                "height": 480
            }
        }).run()
    )


def test_split_passthrough_concat():
    input_video_path = "../files/1min.mp4"
    output_path = "./split_passthrough_concat.mp4"

    # create graph
    my_graph = bmf.graph({
        "dump_graph": 1,
        "graph_name": "split_passthrough_concat"
    })

    # main video
    video = my_graph.decode({'input_path': input_video_path})['video']
    v_l = video.split()
    v1 = v_l[0].pass_through()
    v2 = v_l[1].pass_through()
    v = bmf.concat(v1, v2, n=2, v=1, a=0)
    (
        v.encode(None, {
            "output_path": output_path,
            "video_params": {
                "width": 640,
                "height": 480
            }
        }).run()
    )


def test_several_long_bmf():
        input_path = "../files/1min.mp4"
        output_path = "../output/several_long_bmf.mp4"

        duration = 120
        if input_path == "../files/img.mp4":
            duration = 7

        overlay_option = {
            "dump_graph": 0,
            "source": {
                "start": 0,
                "duration": duration,
                "width": 1280,
                "height": 720
            },
            "overlays": [
                {
                    "start": 0,
                    "duration": duration,
                    "width": 300,
                    "height": 200,
                    "pox_x": 0,
                    "pox_y": 0,
                    "loop": 0,
                    "repeat_last": 1
                }
            ]
        }

        concat_option = {
            "dump_graph": 0,
            "width": 1280,
            "height": 720,
            # if have audio input
            "has_audio": 1,
            "video_list": [
                {
                    "start": 0,
                    "duration": duration,
                    "transition_time": 2,
                    "transition_mode": 1
                },
                {
                    "start": 0,
                    "duration": duration,
                    "transition_time": 2,
                    "transition_mode": 1
                },
                {
                    "start": 0,
                    "duration": duration,
                    "transition_time": 2,
                    "transition_mode": 1
                }
            ]
        }

        # create graph
        my_graph = bmf.graph({
            "dump_graph": 1,
            "graph_name": "several_long_bmf"
        })

        # three logo video
        logo_1 = my_graph.decode({'input_path': "../files/sample_prefix_logo_x.mov"})['video']
        logo_2 = my_graph.decode({'input_path': "../files/sample_prefix_logo_x.mov"})['video']
        logo_3 = my_graph.decode({'input_path': "../files/sample_prefix_logo_x.mov"})['video']

        # three videos
        video1 = my_graph.decode({
            'input_path': input_path,
            # 'split': 1
        })
        video2 = my_graph.decode({
            'input_path': input_path,
            # 'split': 1
        })
        video3 = my_graph.decode({
            'input_path': input_path,
            # 'split': 1
        })

        # do overlay
        overlay_streams = list()
        overlay_streams.append(bmf.module([video1['video'], logo_1], 'video_overlay', overlay_option)[0])
        overlay_streams.append(bmf.module([video2['video'], logo_2], 'video_overlay', overlay_option)[0])
        overlay_streams.append(bmf.module([video3['video'], logo_3], 'video_overlay', overlay_option)[0])

        # do concat
        concat_streams = (
            bmf.module([
                overlay_streams[0],
                overlay_streams[1],
                overlay_streams[2],
                video1['audio'],
                video2['audio'],
                video3['audio']
            ], 'video_concat', concat_option)
        )

        # encode
        (
            bmf.encode(concat_streams[0], concat_streams[1], {
                "output_path": output_path,
                "video_params": {
                    "width": 1280,
                    "height": 720,
                    "codec": "h264",
                    "preset": "veryfast",
                    "crf": "23"
                }
            }).run()
        )


def test_video_audio_concat():
    input_video_path = "../files/1min.mp4"
    output_path = "../files/video_audio_concat.mp4"

    # create graph
    my_graph = bmf.graph({
        "dump_graph":1,
        "graph_name": "video_audio_concat"
    })

    # main video
    video1 = my_graph.decode({'input_path': input_video_path})
    video2 = my_graph.decode({'input_path': input_video_path})
    video3 = my_graph.decode({'input_path': input_video_path})
    v1 = video1['video']
    v2 = video2['video']
    v3 = video3['video']
    a1 = video1['audio']
    a2 = video2['audio']
    a3 = video3['audio']
    v = bmf.concat(v1, v2, v3, n=3, v=1, a=0)
    a = bmf.concat(a1, a2, a3, n=3, v=0, a=1)
    (
        v.encode(a, {
            "output_path": output_path,
            "video_params": {
                "width": 640,
                "height": 480
            }
        }).run()
    )


def test_multi_output_filter():
    input_video_path = "../files/img.mp4"
    output_path = "./multi_output_filter.mp4"

    # create graph
    my_graph = bmf.graph({
        "dump_graph": 1,
        "graph_name": "multi_output_filter"
    })

    video1 = my_graph.decode({'input_path': input_video_path})['video']
    video2 = my_graph.decode({'input_path': input_video_path})['video']

    video1_l = video1.split()
    v11 = video1_l[0]
    v12 = video1_l[1]
    video2_l = video2.split()
    v21 = video2_l[0]
    v22 = video2_l[1]

    v3 = v11.overlay(v21)
    v12 = v12.pass_through()
    v22 = v22.pass_through()
    v4 = v12.concat(v22)
    # v5 = v3.overlay(v4)
    bmf.encode(v3, None, {
        "output_path": output_path,
        "video_params": {
            "width": 1280,
            "height": 720
        }
    })
    bmf.encode(v4, None, {
        "output_path": output_path,
        "video_params": {
            "width": 1280,
            "height": 720
        }
    })
    my_graph.run()


def test_decoder_encoder():
    input_video_path = "../files/1min.mp4"
    output_path = "./decoder_encoder.mp4"

    # create graph
    my_graph = bmf.graph({
        "dump_graph": 1,
        "graph_name": "decoder_encoder"
    })

    video = my_graph.decode({'input_path': input_video_path})

    v = video['video']
    a = video['audio']
    (
        bmf.encode(v, a, {
            "output_path": output_path,
            "video_params": {
                "width": 1280,
                "height": 720
            }
        }).run()
    )


def test_split_fast_slow():
    input_video_path = "../files/1min.mp4"
    output_path = "./split_fast_slow.mp4"
    # create graph
    my_graph = bmf.graph({
        "dump_graph": 1,
        "graph_name": "split_fast_slow",
        "scheduler_count": 4
    })
    video = my_graph.decode({'input_path': input_video_path})['video']
    v_l = video.split()
    v_l.get_node().scheduler_ = 1
    v1 = v_l[0]
    v2 = v_l[1]
    v1 = v1.module("pass_through_fast")
    v1.get_node().scheduler_ = 2
    v2 = v2.module("pass_through_slow")
    v2.get_node().scheduler_ = 3
    my_graph.run()


def test_5_concat():
    bmf.Log.set_log_level(bmf.LogLevel.DEBUG)
    input_video_path = "../files/img.mp4"
    output_path = "./5_concat.mp4"

    # create graph
    my_graph = bmf.graph({
        "dump_graph": 1,
        "graph_name": "5_concat"
    })

    # main video
    video_1 = my_graph.decode({'input_path': input_video_path})

    video_2 = my_graph.decode({'input_path': input_video_path})

    video_1_list = video_1['audio'].asplit()
    video1 = video_1_list[0]
    video2 = video_1_list[1]
    video2 = video2.pass_through()
    # video3 = my_graph.anullsrc(channel_layout='stereo', r=48000, ).atrim(start=0.0, duration=7)
    video3 = my_graph.anullsrc('r=48000').atrim(start=0.0, duration=7)
    # video3 = my_graph.aevalsrc('0:sample_rate=48000:d=7')
    video_2_list = video_2['audio'].asplit()
    video4 = video_2_list[0]
    video5 = video_2_list[1]
    video5 = video5.pass_through()

    concat_video1 = (
        bmf.concat(video1, video2, n=2, v=0, a=1)
    )
    concat_video2 = (
        bmf.concat(video3, video4, n=2, v=0, a=1)
    )
    concat_video = (
        bmf.concat(concat_video1, concat_video2, video5, n=3, v=0, a=1)
    )

    # encode
    (
        bmf.encode(None, concat_video, {
            "output_path": output_path,
            "video_params": {
                "width": 1280,
                "height": 720
            }
        }).run()
    )


if __name__ == '__main__':
    bmf.Log.set_log_level(bmf.LogLevel.DEBUG)
    start = time.time()
    # test_3_concat()
    # test_triangle_concat()
    # test_last_pic()
    # test_split_passthrough_concat()
    # test_several_long_bmf()
    # test_video_audio_concat()
    # test_multi_output_filter()
    # test_decoder_encoder()
    # test_split_fast_slow()
    # test_5_concat()
    end = time.time()
    print("total time is : ", end-start)
