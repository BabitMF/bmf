import bmf
from bmf import SubGraph

'''
Option example:
option = {
    'Type': 'audio',
    'Source': '../files/img.mp4',
    'StartTime': 0,
    'Duration': 5,
    'Trims': [
        [0, 1.2],
        [3, 5.5]
    ],
    'Volume': 0.1,
    'Loop': 0
}
'''


class audio_preprocess(SubGraph):
    def create_graph(self, option=None):
        # create source audio stream for subgraph
        self.inputs.append('source_audio')
        audio_stream = self.graph.input_stream('source_audio')

        # do audio multi-cut
        trims = option.get('Trims')
        if trims and len(trims) > 0:
            # we must copy streams before multi-cut
            source_a_list = audio_stream.ff_filter('asplit', len(trims))
            trims_audio_list = []
            for i in range(len(trims)):
                trim = trims[i]
                start = trim[0]
                end = trim[1]
                a_video = source_a_list[i].ff_filter('atrim', start=start, end=end).asetpts('PTS-STARTPTS')
                trims_audio_list.append(a_video)
            # concat all slice
            audio_stream = bmf.concat(*trims_audio_list, n=len(trims_audio_list), v=0, a=1)

        # audio speed
        speed = option.get('Speed')
        if speed:
            audio_stream = audio_stream.ff_filter('atempo', str(speed))

        # audio duration according to the record
        audio_stream = audio_stream.ff_filter('atrim', start=0, end=option['initial_info']['audio_duration']).asetpts(
            'PTS-STARTPTS')

        # audio volume
        volume = option.get('Volume')
        if volume:
            audio_stream = audio_stream.ff_filter('volume', volume=volume)

        # finish creating graph
        self.output_streams = self.finish_create_graph([audio_stream])
