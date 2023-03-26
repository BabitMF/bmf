import bmf
from bmf import SubGraph

'''
Option example:
    option = {
        "audios": [
            {
                "start": 0,
                "duration": 7
            },
            {
                "start": 2,
                "duration": 5
            }
        ]
    }
'''


class audio_mix(SubGraph):
    def create_graph(self, option=None):
        # create overlay stream
        audio_streams = []
        for (i, _) in enumerate(option['audios']):
            self.inputs.append('audio_' + str(i))
            audio_streams.append(self.graph.input_stream('audio_' + str(i)))

        # overlay processing
        p_audio_streams = []
        for (i, audio_stream) in enumerate(audio_streams):
            info = option['audios'][i]

            # overlay layer pre-processing
            p_audio_stream = (
                audio_stream.atrim(end=info['duration'])
                    .adelay('%d|%d' % (info['start'] * 1000, info['start'] * 1000))
            )
            p_audio_streams.append(p_audio_stream)

        output_stream = (
            bmf.amix(p_audio_streams, inputs=len(p_audio_streams), duration='longest')
        )

        # finish creating graph
        self.output_streams = self.finish_create_graph([output_stream])
