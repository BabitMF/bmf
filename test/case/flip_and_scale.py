import bmf
from bmf import Module, Log, LogLevel, InputType, ProcessResult, Packet, Timestamp, scale_av_pts, av_time_base, \
    BmfCallBackType, SubGraph


class flip_and_scale(SubGraph):

    def create_graph(self, option=None):
        vstreamName = "flip_v_01"
        self.inputs.append(vstreamName)
        v1 = self.graph.input_stream(vstreamName)
        v1 = v1.scale(500, 400).vflip().trim(start=0, end=5).setpts('PTS-STARTPTS')

        astreamName = "flip_a_01"
        self.inputs.append(astreamName)
        a1 = self.graph.input_stream(astreamName)
        a1 = a1.atrim(start=0, end=5).asetpts('PTS-STARTPTS')

        video2 = self.graph.decode({'input_path': "../files/big_bunny_1min_30fps_only_video.mp4"})
        v2 = video2['video']
        v2 = v2.trim(start=30, end=35).setpts('PTS-STARTPTS').scale(500, 400)
        a2 = video2['audio']
        a2 = a2.atrim(start=30, end=35).asetpts('PTS-STARTPTS')

        concat_video = bmf.module([v1, v2], 'simple_video_concat', {"dump_graph": 1})
        concat_audio = bmf.module([a1, a2], 'simple_audio_concat', {"dump_graph": 1})

        # run graph, get output streams
        self.output_streams = self.finish_create_graph([concat_video, concat_audio])
