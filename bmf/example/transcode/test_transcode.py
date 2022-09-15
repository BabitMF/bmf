import sys
import time
import unittest

sys.path.append("../../")
import bmf
import hmp
import timeout_decorator

sys.path.append("../")
from base_test.base_test_case import BaseTestCase
from base_test.media_info import MediaInfo


class TestTranscode(BaseTestCase):
    @timeout_decorator.timeout(seconds=120)
    def test_audio(self):
        input_video_path = "../files/img.mp4"
        output_path = "./audio.mp4"
        expect_result = '../transcode/audio.mp4|0|0|7.617000|MOV,MP4,M4A,3GP,3G2,MJ2|136092|129577||{}'
        self.remove_result_data(output_path)
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
        self.check_video_diff(output_path, expect_result)

    @timeout_decorator.timeout(seconds=120)
    def test_with_input_only_audio(self):
        input_video_path = "../files/only_audio.mp4"
        output_path = "./output.mp4"
        expect_result = '|1080|1920|7.615000|MOV,MP4,M4A,3GP,3G2,MJ2|4483427|4267663|h264|' \
                        '{"fps": "30.0662251656"}'
        self.remove_result_data(output_path)
        mygraph = bmf.graph()
        streams = mygraph.decode({'input_path': input_video_path})
        bmf.encode(None,streams["audio"], {"output_path": output_path}).run()

    @timeout_decorator.timeout(seconds=120)
    def test_with_encode_with_audio_stream_but_no_audio_frame(self):
        input_video_path = "../files/only_video.mp4"
        output_path = "./output.mp4"
        expect_result = '|1080|1920|7.615000|MOV,MP4,M4A,3GP,3G2,MJ2|4483427|4267663|h264|' \
                        '{"fps": "30.0662251656"}'
        self.remove_result_data(output_path)
        mygraph = bmf.graph()
        streams = mygraph.decode({'input_path': input_video_path})
        bmf.encode(streams["video"],streams["audio"], {"output_path": output_path}).run()

    @timeout_decorator.timeout(seconds=120)
    def test_with_null_audio(self):
        input_video_path = "../files/img.mp4"
        output_path = "./with_null_audio.mp4"
        expect_result = '../transcode/with_null_audio.mp4|240|320|7.550000|MOV,MP4,M4A,3GP,3G2,MJ2|238413|225003|h264|' \
                        '{"fps": "30.0662251656"}'
        self.remove_result_data(output_path)
        # create graph
        graph = bmf.graph()

        # decode
        video = graph.decode({
            "input_path": input_video_path
        })

        # create a null audio stream
        audio_stream = graph.anullsrc('r=48000', 'cl=2').atrim('start=0', 'end=6')

        (
            bmf.encode(
                video['video'],
                audio_stream,
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
        self.check_video_diff(output_path, expect_result)

    @timeout_decorator.timeout(seconds=120)
    def test_simple(self):
        input_video_path = "../files/img.mp4"
        output_path = "./simple.mp4"
        expect_result = '../transcode/simple.mp4|240|320|7.617000|MOV,MP4,M4A,3GP,3G2,MJ2|369643|351947|h264|' \
                        '{"fps": "30.0662251656"}'
        self.remove_result_data(output_path)

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
                        "psnr": 1,
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
        self.check_video_diff(output_path, expect_result)

    @timeout_decorator.timeout(seconds=120)
    def test_cb(self):
        input_video_path = "../files/img.mp4"
        output_path = "./cb.mp4"
        expect_result = '../transcode/cb.mp4|240|320|7.615000|MOV,MP4,M4A,3GP,3G2,MJ2|366635|348991|h264|' \
                        '{"fps": "30.0662251656"}'
        self.remove_result_data(output_path)
        # create graph
        graph = bmf.graph()

        def cb(para):
            print(para)
            return bytes("OK","ASCII")

        graph.add_user_callback(bmf.BmfCallBackType.LATEST_TIMESTAMP, cb)

        # decode
        video = graph.decode({
            "input_path": input_video_path
        })

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
        ).node_.add_user_callback(bmf.BmfCallBackType.LATEST_TIMESTAMP,cb)
        graph.run()
        self.check_video_diff(output_path, expect_result)

    @timeout_decorator.timeout(seconds=120)
    def test_hls(self):
        input_video_path = "../files/img.mp4"
        output_path = "./file000.ts"
        expect_result = './transcode/file000.ts|1080|1920|7.616000|MPEGTS|4638189|4415556|h264|' \
                        '{"fps": "29.97"}'
        self.remove_result_data(output_path)
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
                    "output_path": "./out.hls",
                    "format": "hls",
                    "mux_params": {
                        "hls_list_size": "0",
                        "hls_time": "10",
                        "hls_segment_filename": "./file%03d.ts"
                    }
                }
            ).run()
        )
        self.check_video_diff(output_path, expect_result)

    @timeout_decorator.timeout(seconds=120)
    def test_crypt(self):
        input_video_path = "../files/encrypt.mp4"
        output_path = "./crypt.mp4"
        expect_result = '../transcode/crypt.mp4|640|360|10.076000|MOV,MP4,M4A,3GP,3G2,MJ2|991807|1249182|h264|' \
                        '{"fps": "20.0828500414"}'
        self.remove_result_data(output_path)

        # create graph
        graph = bmf.graph()

        # decode
        video = graph.decode({
            "input_path": input_video_path,
            "decryption_key": "b23e92e4d603468c9ec7be7570b16229"
        })

        (
            bmf.encode(
                video['video'],
                video['audio'],
                {
                    "output_path": output_path,
                    "mux_params": {
                        "encryption_scheme": "cenc-aes-ctr",
                        "encryption_key": "76a6c65c5ea762046bd749a2e632ccbb",
                        "encryption_kid": "a7e61c373e219033c21091fa607bf3b8"
                    }
                }
            ).run()
        )
        self.check_video_diff(output_path, expect_result)

    @timeout_decorator.timeout(seconds=120)
    def test_option(self):
        input_video_path = "../files/img.mp4"
        output_path = "./option.mp4"
        expect_result = '../transcode/option.mp4|720|1280|5.643000|MOV,MP4,M4A,3GP,3G2,MJ2|3265125|2303138|h264|' \
                        '{"fps": "30.1796407186"}'
        self.remove_result_data(output_path)
        # create graph
        graph = bmf.graph()

        # decode
        # start_time in seconds
        video = graph.decode({
            "input_path": input_video_path,
            "start_time": 2
        })

        (
            bmf.encode(
                video['video'],
                video['audio'],
                {
                    "output_path": output_path,
                    "video_params": {
                        "codec": "h264",
                        "width": 1280,
                        "height": 720,
                        "preset": "fast",
                        "crf": "23",
                        "x264-params": "ssim=1:psnr=1"
                    },
                    "audio_params": {
                        "codec": "aac",
                        "bit_rate": 128000,
                        "sample_rate": 44100,
                        "channels": 2
                    },
                    "mux_params": {
                        "fflags": "+igndts",
                        "movflags": "+faststart+use_metadata_tags",
                        "max_interleave_delta": "0"
                    }
                }
            ).run()
        )
        self.check_video_diff(output_path, expect_result)

    @timeout_decorator.timeout(seconds=120)
    def test_image(self):
        input_video_path = "../files/overlay.png"
        output_path = "./image.jpg"
        expect_result = 'image.jpg|240|320|0.040000|IMAGE2|975400|4877|mjpeg|{"fps": "0"}'
        self.remove_result_data(output_path)
        (
            bmf.graph()
                .decode({'input_path': input_video_path})['video']
                .scale(320, 240)
                .encode(None, {
                "output_path": output_path,
                "format": "mjpeg",
                "video_params": {
                    "codec": "jpg",
                    "width": 320,
                    "height": 240
                }
            }).run()
        )
        # retrieve log
        self.check_video_diff(output_path, expect_result)

    @timeout_decorator.timeout(seconds=120)
    def test_video(self):
        input_video_path_1 = "../files/header.mp4"
        input_video_path_2 = "../files/header.mp4"
        input_video_path_3 = '../files/img.mp4'
        logo_video_path_1 = "../files/xigua_prefix_logo_x.mov"
        logo_video_path_2 = "../files/xigua_loop_logo2_x.mov"
        output_path = "./video.mp4"
        expect_result = '../transcode/video.mp4|720|1280|13.652000|MOV,MP4,M4A,3GP,3G2,MJ2|3902352|6659364|h264|' \
                        '{"fps": "43.29"}'
        self.remove_result_data(output_path)
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
                               "x264-params": "ssim=1:psnr=1",
                               "vsync": "vfr",
                               "max_fr": 60
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
        self.check_video_diff(output_path, expect_result)

    @timeout_decorator.timeout(seconds=120)
    def test_concat_video_and_audio(self):
        input_video_path = '../files/img.mp4'
        output_path = "./concat_video_and_audio.mp4"
        expect_result = '../transcode/concat_video_and_audio.mp4|1080|1920|15.208000|MOV,MP4,M4A,3GP,3G2,MJ2|4496299|' \
                        '8547466|h264|{"fps": "30.0165289256"}'
        self.remove_result_data(output_path)

        # create graph
        my_graph = bmf.graph()

        # decode video
        video1 = my_graph.decode({'input_path': input_video_path})
        video2 = my_graph.decode({'input_path': input_video_path})

        # do concat
        concat_video_stream = (
            bmf.concat(
                video1['video'],
                video2['video'],
            )
        )

        concat_audio_stream = (
            bmf.concat(
                video1['audio'],
                video2['audio'],
                v=0,
                a=1
            )
        )

        # encode
        (
            bmf.encode(
                concat_video_stream,
                concat_audio_stream,
                {
                    "output_path": output_path
                }
            ).run()
        )
        self.check_video_diff(output_path, expect_result)

    @timeout_decorator.timeout(seconds=120)
    def test_short_video_concat(self):
        input_video_path = "../files/img.mp4"
        input_video_path2 = "../files/single_frame.mp4"

        output_path = "./simple.mp4"
        # create graph
        graph = bmf.graph()

        # create graph
        graph = bmf.graph({'dump_graph': 1})

        # decode
        video = graph.decode({
            "input_path": input_video_path
        })['video']
        video2 = graph.decode({
            "input_path": input_video_path2
        })['video']

        vout = video.concat(video2)

        (
            bmf.encode(
                vout,
                None,
                {
                    "output_path": output_path,
                    "video_params": {
                        "codec": "h264",
                        "width": 320,
                        "height": 240,
                        "crf": 23,
                        "preset": "veryfast"
                    }
                }
            )
                .run()
        )

    @timeout_decorator.timeout(seconds=120)
    def test_map_param(self):
        input_video_path = "../files/multi_stream_video.mp4"
        output_path_1 = "./output_1.mp4"
        output_path_2 = "./output_2.mp4"
        expect_result_1 = '../transcode/output_1.mp4|640|720|7.617000|MOV,MP4,M4A,3GP,3G2,MJ2|1461476|1391317|h264|' \
                        '{"fps": "30.10"}'
        expect_result_2 = '../transcode/output_2.mp4|1080|1920|7.617000|MOV,MP4,M4A,3GP,3G2,MJ2|4390728|4180522|h264|' \
                        '{"fps": "30.10"}'
        self.remove_result_data(output_path_1)
        self.remove_result_data(output_path_2)

        # create graph
        graph = bmf.graph()

        # decode video1
        video1 = graph.decode({
            "input_path": input_video_path,
            "map_v": 0,
            "map_a": 0
        })

        bmf.encode(
            video1['video'],
            video1['audio'],
            {
                "output_path": output_path_1,
            }
        )

        # decode video2
        video2 = graph.decode({
            "input_path": input_video_path,
            "map_v": 1,
            "map_a": 1
        })

        bmf.encode(
            video2['video'],
            video2['audio'],
            {
                "output_path": output_path_2,
            }
        )

        graph.run()

        self.check_video_diff(output_path_1, expect_result_1)
        self.check_video_diff(output_path_2, expect_result_2)

    # @timeout_decorator.timeout(seconds=120)
    # def test_hwaccel_param(self):
    #     graph = bmf.graph()
    #     input_video_path = "../files/img.mp4"
    #     output_path = "./hwaccel.mp4"
    #     expect_result = 'hwaccel.mp4|1080|1920|7.550000|MOV,MP4,M4A,3GP,3G2,MJ2|13588330|12823987|h264|{"fps": "30.066225165562916"}'
    #     video = graph.decode({
    #         "input_path": input_video_path,
    #         "autorotate":0,
    #         "video_params":{"hwaccel":"cuda"}
    #     })

    #     (
    #         bmf.encode(
    #             video['video'],
    #             None,
    #             {
    #                 "output_path": output_path,
    #                 "video_params": {
    #                     "max_fr": 60,
    #                     "codec": "h264_nvenc",
    #                     "pix_fmt":"cuda"
    #                 },
    #                 "audio_params": {
    #                     "codec": "aac",
    #                     "bit_rate": 128000,
    #                     "sample_rate": 44100,
    #                     "channels": 2
    #                 }
    #             }
    #         ).run()
    #     )
    #     self.check_video_diff(output_path, expect_result)

    @timeout_decorator.timeout(seconds=120)
    def test_rgb_2_video(self):
        input_path = "../files/test_rgba_806x654.rgb"
        output_path = "./rgb2video.mp4"
        expect_result = '../transcode/rgb2video.mp4|654|806|2.04|MOV,MP4,M4A,3GP,3G2,MJ2|58848|15014|h264|' \
                        '{"fps": "25.0"}'
        stream = bmf.graph().decode({
            'input_path': input_path,
            's': "806:654",
            'pix_fmt': "rgba"
        })

        video_stream = stream['video'].ff_filter('loop', loop=50, size=1)

        video_stream.encode(None, {
            "output_path": output_path
        }).run()

        self.check_video_diff(output_path, expect_result)

    @timeout_decorator.timeout(seconds=120)
    def test_stream_copy(self):
        input_path = "../files/img.mp4"
        output_path = "./stream_copy.mp4"
        expect_result = './transcode/stream_copy.mp4|1080|1920|7.616000|MOV,MP4,M4A,3GP,3G2,MJ2|4638189|4415556|mpeg4|' \
                        '{"fps": "29.97"}'
        stream = bmf.graph().decode({
            'input_path': "../files/img.mp4",
            'video_codec': "copy"
        })

        video_stream = stream['video']

        video_stream.encode(stream['audio'], {
            "output_path": "stream_copy.mp4",
        }).run()

        self.check_video_diff(output_path, expect_result)

    @timeout_decorator.timeout(seconds=120)
    def test_stream_audio_copy(self):
        input_path = "../files/live_audio.flv"
        output_path = "./audio_copy.mp4"
        expect_result = './transcode/audio_copy.mp4|0|0|100.21|MOV,MP4,M4A,3GP,3G2,MJ2|132133|1658880||{"accurate": "b"}' # accurate could be "b" bitrate accurate check, "d" duration accurate check and "f" fps accurate check
        stream = bmf.graph().decode(
                             {
                                'input_path': input_path,
                                'video_codec': "copy",
                                'audio_codec': "copy"
                             })

        (
        bmf.encode(None, stream['audio'],
                  {
                    "output_path": output_path
                  }
                  ).run()
        )

        self.check_video_diff(output_path, expect_result)
    
    @timeout_decorator.timeout(seconds=120)
    def test_extract_frames(self):
        input_path = "../files/img.mp4"
        frames = (
                bmf.graph()
                    .decode({'input_path': "../files/img.mp4", "video_params":{"extract_frames":{ "fps":0.5}} })['video']
                    .start()  # this will return a packet generator
            )
        
        num = 0
        for i, frame in enumerate(frames):
            # convert frame to a nd array
            if frame is not None:
                num = num+1

                # we can add some more processing here, e.g. predicting
            else:
                break
        assert(num == 5)

    @timeout_decorator.timeout(seconds=120)
    def test_incorrect_stream_notify(self):
        output_path = "./incorrect_stream_notify.mp4"

        video = bmf.graph().decode({
            "input_path": "../files/img.mp4",
        })

        stream_notify = 'wrong_name'
        v = video[stream_notify]

        try:
            bmf.encode(v, None,
                {
                    "output_path": output_path
                }
            ).run()
        except Exception as e:
            print(e)

    @timeout_decorator.timeout(seconds=120)
    def test_incorrect_encoder_param(self):
        output_path = "./incorrect_encoder_param.mp4"

        video = bmf.graph().decode({
            "input_path": "../files/img.mp4",
        })

        v = video['video']
        a = video['audio']

        wrong_k_1 = 'wrong_key_1'
        wrong_v_1 = 'wrong_value_1'
        wrong_k_2 = 'wrong_key_2'
        wrong_v_2 = 'wrong_value_2'

        try:
            bmf.encode(v, a,
                {
                    "output_path": output_path,
                    "video_params": {
                        "codec": "h264",
                        "preset": "fast",
                        "crf": "23",
                        wrong_k_1: wrong_v_1,
                        wrong_k_2: wrong_v_2
                    },
                    "audio_params": {
                        wrong_k_1: wrong_v_1,
                        wrong_k_2: wrong_v_2
                    },
                    "mux_params": {
                        wrong_k_1: wrong_v_1,
                        wrong_k_2: wrong_v_2
                    }
                }
            ).run()
        except Exception as e:
            print(e)

    @timeout_decorator.timeout(seconds=120)
    def test_durations(self):
        input_video_path = "../files/img.mp4"
        output_path = "./durations.mp4"
        expect_result = '../transcode/durations.mp4|240|320|4.54|MOV,MP4,M4A,3GP,3G2,MJ2|255975|147456|h264|' \
                        '{"fps": "16.67"}'
        graph = bmf.graph()
        graph = bmf.graph({'dump_graph':1})

        video = graph.decode({
            "input_path": input_video_path,
            "durations": [1.5, 3, 5, 6]
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
                        "preset": "veryfast",
                        "vsync": "vfr",
                        "r": 30
                    }
                }
            )
            .run()
        )
        self.check_video_diff(output_path, expect_result)

    @timeout_decorator.timeout(seconds=120)
    def test_output_raw_video(self):
        input_video_path = '../files/img.mp4'
        raw_output_path = "./out.yuv"

        my_graph = bmf.graph()
        video1 = my_graph.decode({'input_path': input_video_path})

        # get raw data of the video stream
        raw_output = (
            bmf.encode(
                video1['video'],
                None,
                {
                    "video_params": {
                        "codec": "rawvideo",
                    },
                    "format": "rawvideo",
                    "output_path": raw_output_path
                }
            )
        ).run()

        self.check_md(raw_output_path, "adba294376cfcb454603a44990a7d7bc")

    @timeout_decorator.timeout(seconds=120)
    def test_output_null(self):
        input_video_path = "../files/img.mp4"

        graph = bmf.graph()

        video = graph.decode({
            "input_path": input_video_path
        })

        (
            bmf.encode(
                video['video'],
                video['audio'],
                {
                    "null_output": 1
                }
            )
            .run()
        )

    @timeout_decorator.timeout(seconds=120)
    def test_vframes(self):
        input_video_path = "../files/img.mp4"
        output_path = "./simple.mp4"
        expect_result = './transcode/simple.mp4|480|640|1.001000|MOV,MP4,M4A,3GP,3G2,MJ2|822489|102914|h264|' \
       '{"fps": "29.97"}'
        graph = bmf.graph({'dump_graph':1})
        video = graph.decode({
            "input_path": input_video_path,
            #"vframes": 30
        })
        (
            bmf.encode(
                video['video'],
                None,
                {
                    "output_path": output_path,
                    "vframes": 30,
                    "video_params": {
                        "codec": "h264",
                        "width": 640,
                        "height": 480,
                        "crf": 23,
                        "preset": "veryfast"
                    },
                }
            )
            .run()
        )
        self.check_video_diff(output_path, expect_result)

    @timeout_decorator.timeout(seconds=120)
    def test_segment_trans(self):
        input_video_path = "../files/img.mp4"
        output_path = "./simple_%05d.mp4"
        expect_result1 = './transcode/simple_00000.mp4|1080|1920|4.041000|MOV,MP4,M4A,3GP,3G2,MJ2|4385223|2215086|mpeg4|' \
       '{"fps": "29.97", "accurate": "d"}'
        expect_result2 = './transcode/simple_00001.mp4|1080|1920|3.576000|MOV,MP4,M4A,3GP,3G2,MJ2|3861794|1726222|mpeg4|' \
       '{"fps": "29.97", "accurate": "d"}'

        graph = bmf.graph({'dump_graph':1})
        video = graph.decode({
            "input_path": input_video_path,
            'video_codec': "copy"
        })
        (
            bmf.encode(
                video['video'],
                video['audio'],
                {
                    "output_path": output_path,
                    "format": "segment",
                    "mux_params": {
                        "segment_time": 4
                    },
                }
            )
            .run()
        )
        self.check_video_diff('./simple_00000.mp4', expect_result1)
        self.check_video_diff('./simple_00001.mp4', expect_result2)

    @timeout_decorator.timeout(seconds=120)
    def test_encoder_push_output_mp4(self):
        input_video_path = "../files/img.mp4"
        output_path = "./simple_vframe_python.mp4"
        graph = bmf.graph({'dump_graph':1})
        video = graph.decode({
            "input_path": input_video_path,
        })
        result = (
            bmf.encode(
                video['video'],
                None,
                {
                    "output_path": output_path,
                    "push_output": 1,
                    "vframes": 60,
                    "video_params": {
                        "codec": "jpg",
                        "width": 640,
                        "height": 480,
                        "crf": 23,
                        "preset": "veryfast"
                    },
                }
            )
            .start()
        )
        with open(output_path, "wb") as f:
            for i, packet in enumerate(result):
                avpacket = packet.get(bmf.BMFAVPacket)
                offset = avpacket.get_offset()
                whence = avpacket.get_whence()
                """
                offset is file write pointer offset, whence is mode. whence == SEEK_SET, from begin; whence == SEEK_CUR, current;
                whence == SEEK_END, from end, etc.
                #define SEEK_SET	0	/* Seek from beginning of file.  */
                #define SEEK_CUR	1	/* Seek from current position.  */
                #define SEEK_END	2	/* Seek from end of file.  */
                #define SEEK_DATA	3	/* Seek to next data.  */
                #define SEEK_HOLE	4	/* Seek to next hole.  */

                NOTICE: BMFAVPacket's data is uint8_t pointer, but sdk_py_get_data convert pointer into char pointer
                so the result is wrong. please use cplusplus interface to instead.

                boost::python::object sdk_py_BMFAVPacket::sdk_py_get_data() {
                    void *data = this->get_data();
                    boost::python::object
                    pkt_data(boost::python::handle<>(PyMemoryView_FromMemory((char *)data, this->get_size(),
                    PyBUF_WRITE)));
                    return pkt_data;
                }
                """
                data = avpacket.data.numpy()
                if (offset > 0) :
                    f.seek(offset, whence)
                f.write(data)

    @timeout_decorator.timeout(seconds=120)
    def test_encoder_push_output_image2pipe(self):
        input_video_path = "../files/img.mp4"
        graph = bmf.graph({'dump_graph':1})
        video = graph.decode({
            "input_path": input_video_path,
        })
        vframes_num = 2
        result = (
            bmf.encode(
                video['video'],
                None,
                {
                    "push_output": 1,
                    "vframes": vframes_num,
                    "format": "image2pipe",
                    "avio_buffer_size": 65536,#16*4096
                    "video_params": {
                        "codec": "jpg",
                        "width": 640,
                        "height": 480,
                        "crf": 23,
                        "preset": "veryfast"
                    },
                }
            )
            .start()
        )
        write_num = 0
        for i, packet in enumerate(result):
            avpacket = packet.get(bmf.BMFAVPacket)
            data = avpacket.data.numpy()
            if write_num < vframes_num:
                output_path = "./simple_image" + str(write_num)+ ".jpg"
                write_num = write_num + 1
                with open(output_path, "wb") as f:
                    f.write(data)

    @timeout_decorator.timeout(seconds=120)
    def test_encoder_push_output_audio_pcm_s16le(self):
        input_video_path = "../files/img.mp4"
        output_path = "./test_audio_simple_pcm_s16le.wav"
        graph = bmf.graph({'dump_graph':1})
        video = graph.decode({
            "input_path": input_video_path,
            #'audio_codec': "copy",
        })
        result = (
            bmf.encode(
                None,
                video['audio'],
                {
                    "output_path": output_path,
                    "format": "wav",
                    "push_output": 1,
                    "audio_params": {
                        "codec": "pcm_s16le",
                    },
                }
            )
            .start()
        )
        with open(output_path, "wb") as f:
            for i, packet in enumerate(result):
                avpacket = packet.get(bmf.BMFAVPacket)
                data = avpacket.data.numpy()
                offset = avpacket.get_offset()
                whence = avpacket.get_whence()
                if offset > 0:
                    f.seek(offset, whence)
                f.write(data)

    @timeout_decorator.timeout(seconds=120)
    def test_skip_frame(self):
        input_video_path = "../files/img.mp4"
        output_path = "./test_skip_frame_videp.mp4"
        expect_result = '../transcode/test_skip_frame_videp.mp4|1080|1920|7.574233|MOV,MP4,M4A,3GP,3G2,MJ2|1321859|1255038|h264|' \
            '{"fps": "29.97"}'
        # 创建BMF Graph
        graph = bmf.graph()
    
        # 构建解码流
        streams = graph.decode({
            "input_path": input_video_path,
            "skip_frame" : 32
        })

        (   
            bmf.encode(
                streams['video'],
                None,
                {
                    "output_path": output_path,
                    "video_params": {
                        "codec": "h264",
                        "crf": 23,
                        "preset": "veryfast",
                    }
                }
            )
            .run()
        )
        self.check_video_diff(output_path, expect_result)

if __name__ == '__main__':
    unittest.main()
