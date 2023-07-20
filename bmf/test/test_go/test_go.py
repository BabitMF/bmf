import sys
import unittest
import os

import timeout_decorator

sys.path.append("../")
from base_test.base_test_case import BaseTestCase
from base_test.media_info import MediaInfo

# expect_result = '../transcode/audio.mp4|0|0|7.617000|MOV,MP4,M4A,3GP,3G2,MJ2|136092|129577||{}'
# BaseTestCase.check_video_diff("","","")

class TestGO(BaseTestCase):
    @timeout_decorator.timeout(seconds=120)
    def test_normal_mode(self):
        output_path = "./output_normal_mode.mp4"
        expect_result = './output_normal_mode.mp4|1080|1920|10.008|MOV,MP4,M4A,3GP,3G2,MJ2|1918874|2400512|h264|{"fps": "29.97"}'
        self.remove_result_data(output_path)
        os.system("./main normalMode")
        self.check_video_diff(output_path, expect_result)

    @timeout_decorator.timeout(seconds=120)
    def test_audio(self):
        output_path = "./audio.mp4"
        expect_result = './audio.mp4|0|0|10.008|MOV,MP4,M4A,3GP,3G2,MJ2|136092|166183||{}'
        self.remove_result_data(output_path)
        os.system("./main testAudio")
        self.check_video_diff(output_path, expect_result)

    def test_with_input_only_audio(self):
        output_path = "./output.mp4"
        expect_result = '|0|0|10.008|MOV,MP4,M4A,3GP,3G2,MJ2|132840|166183||{}'        
        self.remove_result_data(output_path)
        os.system("./main testWithInputOnlyAudio")
        self.check_video_diff(output_path, expect_result)
        
    @timeout_decorator.timeout(seconds=120)
    def test_with_encode_with_audio_stream_but_no_audio_frame(self):
        output_path = "./output.mp4"
        expect_result = '|1080|1920|10.0|MOV,MP4,M4A,3GP,3G2,MJ2|1783292|2229115|h264|' \
                        '{"fps": "30.0"}'
        self.remove_result_data(output_path)
        os.system("./main testWithEncodeWithAudioStreamButNoAudioFrame")
        self.check_video_diff(output_path, expect_result)

    @timeout_decorator.timeout(seconds=120)
    def test_with_null_audio(self):
        output_path = "./with_null_audio.mp4"
        expect_result = '../transcode/with_null_audio.mp4|240|320|10.0|MOV,MP4,M4A,3GP,3G2,MJ2|60438|75548|h264|' \
                        '{"fps": "30.0662251656"}'
        self.remove_result_data(output_path)
        os.system("./main testWithNullAudio")
        self.check_video_diff(output_path, expect_result)

    @timeout_decorator.timeout(seconds=120)
    def test_simple(self):
        output_path = "./simple.mp4"
        expect_result = '../transcode/simple.mp4|240|320|10.008|MOV,MP4,M4A,3GP,3G2,MJ2|192235|240486|h264|' \
                        '{"fps": "30.0662251656"}'
        self.remove_result_data(output_path)
        os.system("./main testSimple")
        self.check_video_diff(output_path, expect_result)

    @timeout_decorator.timeout(seconds=120)
    def test_hls(self):
        output_path = "./file000.ts"
        expect_result = './transcode/file000.ts|1080|1920|10.0304|MPEGTS|2029494|2544580|h264|' \
                        '{"fps": "29.97"}'
        self.remove_result_data(output_path)
        os.system("./main testhls")
        self.check_video_diff(output_path, expect_result)

    @timeout_decorator.timeout(seconds=120)
    def test_crypt(self):
        output_path = "./crypt.mp4"
        expect_result = '../transcode/crypt.mp4|640|360|10.076000|MOV,MP4,M4A,3GP,3G2,MJ2|991807|1249182|h264|' \
                        '{"fps": "20.0828500414"}'
        self.remove_result_data(output_path)
        os.system("./main testCrypt")
        self.check_video_diff(output_path, expect_result)

    @timeout_decorator.timeout(seconds=120)
    def test_option(self):
        output_path = "./option.mp4"
        expect_result = '../transcode/option.mp4|720|1280|8.008|MOV,MP4,M4A,3GP,3G2,MJ2|762463|763226|h264|' \
                        '{"fps": "30.1796407186"}'
        self.remove_result_data(output_path)
        os.system("./main testOption")
        self.check_video_diff(output_path, expect_result)

    @timeout_decorator.timeout(seconds=120)
    def test_image(self):
        output_path = "./image.jpg"
        expect_result = 'image.jpg|240|320|0.040000|IMAGE2|975400|4877|mjpeg|{"fps": "25.0"}'
        self.remove_result_data(output_path)
        os.system("./main testImage")
        self.check_video_diff(output_path, expect_result)

    @timeout_decorator.timeout(seconds=120)
    def test_video(self):
        output_path = "./video.mp4"
        expect_result = '../transcode/video.mp4|720|1280|16.045|MOV,MP4,M4A,3GP,3G2,MJ2|2504766|5023622|h264|' \
                        '{"fps": "43.29"}'
        self.remove_result_data(output_path)
        os.system("./main testVideo")
        self.check_video_diff(output_path, expect_result)

    @timeout_decorator.timeout(seconds=120)
    def test_concat_video_and_audio(self):
        output_path = "./concat_video_and_audio.mp4"
        expect_result = '../transcode/concat_video_and_audio.mp4|1080|1920|20.015|MOV,MP4,M4A,3GP,3G2,MJ2|1919964|' \
                        '4803511|h264|{"fps": "30.0165289256"}'
        self.remove_result_data(output_path)
        os.system("./main testConcatVideoAndAudio")
        self.check_video_diff(output_path, expect_result)

    @timeout_decorator.timeout(seconds=120)
    def test_short_video_concat(self):
        output_path = "./concat_video_and_audio.mp4"
        os.system("./main testShortVideoConcat")

    @timeout_decorator.timeout(seconds=120)
    def test_map_param(self):
        output_path_1 = "./output_1.mp4"
        output_path_2 = "./output_2.mp4"
        expect_result_1 = '../transcode/output_1.mp4|720|1280|10.008|MOV,MP4,M4A,3GP,3G2,MJ2|828031|1035868|h264|' \
                        '{"fps": "30.10"}'
        expect_result_2 = '../transcode/output_2.mp4|1080|1920|10.008|MOV,MP4,M4A,3GP,3G2,MJ2|1822167|2283859|h264|' \
                        '{"fps": "30.10"}'
        self.remove_result_data(output_path_1)
        self.remove_result_data(output_path_2)
        os.system("./main testMapParam")
        self.check_video_diff(output_path_1, expect_result_1)
        self.check_video_diff(output_path_2, expect_result_2)

    @timeout_decorator.timeout(seconds=120)
    def test_rgb_2_video(self):
        output_path = "./rgb2video.mp4"
        expect_result = '../transcode/rgb2video.mp4|654|806|2.04|MOV,MP4,M4A,3GP,3G2,MJ2|58848|15014|h264|' \
                        '{"fps": "25.0"}'
        self.remove_result_data(output_path)
        os.system("./main testRGB2Video")
        self.check_video_diff(output_path, expect_result)

    @timeout_decorator.timeout(seconds=120)
    def test_stream_copy(self):
        output_path = "./stream_copy.mp4"
        expect_result = './transcode/stream_copy.mp4|1080|1920|10.008|MOV,MP4,M4A,3GP,3G2,MJ2|2255869|2822093|mpeg4|' \
                        '{"fps": "29.97"}'
        self.remove_result_data(output_path)
        os.system("./main testStreamCopy")
        self.check_video_diff(output_path, expect_result)

    @timeout_decorator.timeout(seconds=120)
    def test_stream_audio_copy(self):
        output_path = "./audio_copy.mp4"
        expect_result = './transcode/audio_copy.mp4|0|0|10.031|MOV,MP4,M4A,3GP,3G2,MJ2|129999|163003||{"accurate": "b"}' # accurate could be "b" bitrate accurate check, "d" duration accurate check and "f" fps accurate check
        self.remove_result_data(output_path)
        os.system("./main testStreamAudioCopy")
        self.check_video_diff(output_path, expect_result)

    @timeout_decorator.timeout(seconds=120)
    def test_extract_frames(self):
        os.system("./main testExtractFrames")

    @timeout_decorator.timeout(seconds=120)
    def test_incorrect_stream_notify(self):
        try:
            os.system("./main testIncorrectStreamNotify")
        except Exception as e:
            print(e)

    @timeout_decorator.timeout(seconds=120)
    def test_incorrect_encoder_param(self):
        try:
            os.system("./main testIncorrectEncoderParam")
        except Exception as e:
            print(e)

    @timeout_decorator.timeout(seconds=120)
    def test_durations(self):
        output_path = "./durations.mp4"
        expect_result = '../transcode/durations.mp4|240|320|4.54|MOV,MP4,M4A,3GP,3G2,MJ2|105089|59113|h264|' \
                        '{"fps": "16.67"}'
        self.remove_result_data(output_path)
        os.system("./main testDurations")
        self.check_video_diff(output_path, expect_result)

    @timeout_decorator.timeout(seconds=120)
    def test_output_raw_video(self):
        raw_output_path = "./out.yuv"
        os.system("./main testOutputRawVideo")
        self.check_md(raw_output_path, "992f929388f18c43c06c767d63eea15d")

    @timeout_decorator.timeout(seconds=120)
    def test_output_null(self):
        os.system("./main testOutputNull")

    @timeout_decorator.timeout(seconds=120)
    def test_vframes(self):
        output_path = "./simple.mp4"
        expect_result = './transcode/simple.mp4|480|640|1.001000|MOV,MP4,M4A,3GP,3G2,MJ2|110976|13872|h264|' \
       '{"fps": "29.97"}'
        self.remove_result_data(output_path)
        os.system("./main testVframes")
        self.check_video_diff(output_path, expect_result)

    @timeout_decorator.timeout(seconds=120)
    def test_segment_trans(self):
        output_path_1 = "./simple_00000.mp4"
        output_path_2 = "./simple_00001.mp4"
        expect_result1 = './transcode/simple_00000.mp4|1080|1920|4.296|MOV,MP4,M4A,3GP,3G2,MJ2|1988878|1068028|mpeg4|' \
       '{"fps": "29.97", "accurate": "d"}'
        expect_result2 = './transcode/simple_00001.mp4|1080|1920|8.313|MOV,MP4,M4A,3GP,3G2,MJ2|1102862|1146012|mpeg4|' \
       '{"fps": "29.97", "accurate": "d"}'
        self.remove_result_data(output_path_1)
        self.remove_result_data(output_path_2)
        os.system("./main testSegmentTrans")
        self.check_video_diff(output_path_1, expect_result1)
        self.check_video_diff(output_path_2, expect_result2)

    @timeout_decorator.timeout(seconds=120)
    def test_encoder_push_output_mp4(self):
        os.system("./main testEncoderPushOutputMp4")

    @timeout_decorator.timeout(seconds=120)
    def test_encoder_push_output_image2pipe(self):
        os.system("./main testEncoderPushOutputImage2Pipe")

    @timeout_decorator.timeout(seconds=120)
    def test_encoder_push_output_audio_pcm_s16le(self):
        os.system("./main testEncoderPushOutputAudioPcmS16le")

    @timeout_decorator.timeout(seconds=120)
    def test_skip_frame(self):
        output_path = "./test_skip_frame_video.mp4"
        expect_result = '../transcode/test_skip_frame_video.mp4|1080|1920|4.3|MOV,MP4,M4A,3GP,3G2,MJ2|22532|12111|h264|' \
            '{"fps": "29.97"}'
        self.remove_result_data(output_path)
        os.system("./main testSkipFrame")
        self.check_video_diff(output_path, expect_result)

    @timeout_decorator.timeout(seconds=120)
    def test_audio_c_module(self):
        output_path = "./audio_c_module.mp4"
        expect_result = 'audio_c_module.mp4|0|0|10.008|MOV,MP4,M4A,3GP,3G2,MJ2|132840|166183||{}'
        self.remove_result_data(output_path)
        os.system("./main testAudioModule")
        self.check_video_diff(output_path, expect_result)

    @timeout_decorator.timeout(seconds=120)
    def test_exception_in_python_module(self):
        output_path = "./test_exception_in_python_module.mp4"
        try:
            os.system("./main testExceptionInPythonModule")
        except Exception as e:
            print(e)

    @timeout_decorator.timeout(seconds=120)
    def test_video_overlays(self):
        output_path = "./overlays.mp4"
        expect_result = '../edit/overlays.mp4|480|640|10.008|MOV,MP4,M4A,3GP,3G2,MJ2|230526|288389|h264|' \
                        '{"fps": "30.0715990453"}'
        self.remove_result_data(output_path)
        os.system("./main testVideoOverlays")
        self.check_video_diff(output_path, expect_result)

    @timeout_decorator.timeout(seconds=120)
    def test_video_concat(self):
        output_path = "./video_concat.mp4"
        expect_result = '../edit/video_concat.mp4|720|1280|15.022000|MOV,MP4,M4A,3GP,3G2,MJ2|385322|722480|h264|' \
                        '{"fps": "30.0166759311"}'
        self.remove_result_data(output_path)
        os.system("./main testVideoConcat")
        self.check_video_diff(output_path, expect_result)

    @timeout_decorator.timeout(seconds=120)
    def test_premodule_mode(self):
        output_path = "./output_pre_module.mp4"
        expect_result = './output_pre_module.mp4|1080|1920|10.0|MOV,MP4,M4A,3GP,3G2,MJ2|1918874|2400512|h264|{"fps": "29.97"}'
        self.remove_result_data(output_path)
        os.system("./main premoduleMode")
        self.check_video_diff(output_path, expect_result)

    @timeout_decorator.timeout(seconds=120)
    def test_sync_mode(self):
        output_path = "./output_sync_mode.mp4"
        expect_result = './output_sync_mode.mp4|1080|1920|10.0|MOV,MP4,M4A,3GP,3G2,MJ2|1783292|2229115|h264|{"fps": "29.97"}'
        self.remove_result_data(output_path)
        os.system("./main syncMode")
        self.check_video_diff(output_path, expect_result)

    @timeout_decorator.timeout(seconds=120)
    def test_sync_mode_serial(self):
        output_path = "./output_sync_mode_serial.mp4"
        expect_result = './output_sync_mode_serial.mp4|1080|1920|10.0|MOV,MP4,M4A,3GP,3G2,MJ2|1783292|2229115|h264|{"fps": "29.97"}'
        self.remove_result_data(output_path)
        os.system("./main syncModeSerial")
        self.check_video_diff(output_path, expect_result)


if __name__ == '__main__':
    unittest.main()
