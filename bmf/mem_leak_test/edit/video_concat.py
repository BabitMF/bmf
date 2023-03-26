import bmf
from bmf import SubGraph

'''
Option example:
    option = {
        "width": 640,
        "height": 480,
        "has_audio": 1,
        "video_list": [
            {
                "start": 0,
                "duration": 2,
                "transition_time": 1,
                "transition_mode": 1
            },
            {
                "start": 0,
                "duration": 4,
                "transition_time": 1,
                "transition_mode": 1
            },
            {
                "start": 3,
                "duration": 4,
                "transition_time": 1,
                "transition_mode": 1
            }
        ]
    }
'''


class video_concat(SubGraph):
    def create_graph(self, option=None):
        video_stream_cnt = len(option['video_list'])

        # here we assume if have audio, audio stream count is equal to video
        if option['has_audio'] == 1:
            audio_stream_cnt = video_stream_cnt
        else:
            audio_stream_cnt = 0

        # process video streams
        concat_video_streams = []
        prev_transition_stream = None
        for i in range(video_stream_cnt):
            # create a input stream
            stream_name = 'video_' + str(i)
            self.inputs.append(stream_name)
            video_stream = (
                self.graph.input_stream(stream_name)
                    .scale(option['width'], option['height'])
            )

            if option['video_list'][i]['transition_time'] > 0 and i < video_stream_cnt - 1:
                split_stream = video_stream.split()
                video_stream = split_stream[0]
                transition_stream = split_stream[1]
            else:
                transition_stream = None

            # prepare concat stream
            info = option['video_list'][i]
            trim_time = info['duration'] - info['transition_time']
            concat_stream = (
                video_stream.trim(start=info['start'], duration=trim_time)
                    .setpts('PTS-STARTPTS')
            )

            # do transition, here use overlay instead
            if prev_transition_stream is not None:
                concat_stream = concat_stream.overlay(prev_transition_stream, repeatlast=0)

            # add to concat stream
            concat_video_streams.append(concat_stream)

            # prepare transition stream for next stream
            if transition_stream is not None:
                prev_transition_stream = (
                    transition_stream.trim(start=trim_time, duration=info['transition_time'])
                        .setpts('PTS-STARTPTS')
                        .scale(200, 200)
                )

        # concat videos
        concat_video_stream = bmf.concat(*concat_video_streams, n=video_stream_cnt, v=1, a=0)

        # process audio
        # actually, we can use another sub-graph module to process audio, we combine it
        # in one module to show how to process multi-output in sub-graph
        concat_audio_stream = None
        if audio_stream_cnt > 0:
            concat_audio_streams = []
            for i in range(audio_stream_cnt):
                # create a input stream
                stream_name = 'audio_' + str(i)
                self.inputs.append(stream_name)

                # pre-processing for audio stream
                info = option['video_list'][i]
                trim_time = info['duration'] - info['transition_time']
                audio_stream = (
                    self.graph.input_stream(stream_name)
                        .atrim(start=info['start'], duration=trim_time)
                        .asetpts('PTS-STARTPTS')
                        .afade(t='in', st=0, d=2)
                        .afade(t='out', st=info['duration'] - 2, d=2)
                )

                # add to concat stream
                concat_audio_streams.append(audio_stream)

            # concat audio
            concat_audio_stream = bmf.concat(*concat_audio_streams, n=audio_stream_cnt, v=0, a=1)

        # finish creating graph
        self.output_streams = self.finish_create_graph([concat_video_stream, concat_audio_stream])
