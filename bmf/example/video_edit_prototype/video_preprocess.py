from bmf import SubGraph
import bmf

'''
Option example:
option = {
    'Type': 'video',
    'Source': '../files/img.mp4',
    'StartTime': 0,
    'Duration': 5,
    'Position': {
        'PosX': '2%',
        'PosY': '2%',
        'Width': '450',
        'Height': '270'
    },
    'Crop': {
        'PosX': '2%',
        'PosY': '2%',
        'Width': '400',
        'Height': '200'
    },
    'Speed': 2.1,
    'Rotate': 90,
    'Delogo': {
        'PosX': '2%',
        'PosY': '2%',
        'Width': '400',
        'Height': '200'
    },
    'Trims': [
        [0, 1.2],
        [3, 5.5]
    ],
    "Filters": {
        "Contrast": 110,
        "Brightness": 62,
        "Saturate": 121,
        "Opacity": 66,
        "Blur": 28
    },
    'ExtraFilters': [
        {
            'Type': 'AAA',
            'Para': 'BBB',
        }
    ],
    'Volume': 0.1,
    'Mute': 0
}
'''


class video_preprocess(SubGraph):
    def create_graph(self, option=None):
        # create source video stream for subgraph
        self.inputs.append('source_video')
        video_stream = self.graph.input_stream('source_video')

        # do video multi-cut
        trims = option.get('Trims')
        if trims and len(trims) > 0:
            # we must copy streams before multi-cut
            source_v_list = video_stream.split(len(trims))
            trims_video_list = []
            for i in range(len(trims)):
                trim = trims[i]
                start = float(trim[0])
                end = float(trim[1])
                v_video = source_v_list[i].ff_filter('trim', start=start, end=end).setpts('PTS-STARTPTS')
                trims_video_list.append(v_video)
            # concat all slice
            video_stream = bmf.concat(*trims_video_list, n=len(trims_video_list), v=1, a=0)

        # video speed
        speed = option.get('Speed')
        if speed:
            video_stream = video_stream.setpts(str(1.0 / float(speed)) + '*PTS')

        # video duration according to the record
        video_stream = video_stream.ff_filter('trim', start=0, end=option['initial_info']['video_duration']).setpts(
            'PTS-STARTPTS')

        # do scale
        position = option.get('Position')
        if position:
            video_stream = video_stream.scale(position['Width'], position['Height'])

        # do crop
        crop = option.get('Crop')
        if crop:
            video_stream = video_stream.ff_filter('crop', crop['Width'], crop['Height'], crop['PosX'], crop['PosY'])

        # do basic filters
        filters = option.get('Filters')
        if filters:
            video_stream = video_stream.ff_filter('eq', contrast=filters['Contrast'],
                                                  brightness=filters['Brightness'], saturation=filters['Saturate'])

        # has audio_stream
        if 'audio_duration' in option['initial_info']:
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
            audio_stream = audio_stream.ff_filter('atrim', start=0,
                                                  end=option['initial_info']['audio_duration']).asetpts('PTS-STARTPTS')

            # audio volume
            volume = option.get('Volume')
            if volume:
                audio_stream = audio_stream.ff_filter('volume', volume=volume)

            # finish creating graph
            self.output_streams = self.finish_create_graph([video_stream, audio_stream])
        else:
            # finish creating graph, with no audio stream
            self.output_streams = self.finish_create_graph([video_stream])
