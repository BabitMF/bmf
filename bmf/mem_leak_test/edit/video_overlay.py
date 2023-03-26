import bmf
from bmf import SubGraph

'''
Option example:
    option = {
        "source": {
            "start": 0,
            "duration": 5,
            "width": 640,
            "height": 480
        },
        "overlays": [
            {
                "start": 0,
                "duration": 2,
                "width": 300,
                "height": 200,
                "pox_x": 0,
                "pox_y": 0,
                "loop": -1,
                "repeat_last": 0
            },
            {
                "start": 2,
                "duration": 2,
                "width": 300,
                "height": 200,
                "pox_x": 'W-300',
                "pox_y": 0,
                "loop": 0,
                "repeat_last": 1
            }
        ]
    }
'''


class video_overlay(SubGraph):
    def create_graph(self, option=None):
        # create source stream
        self.inputs.append('source')
        source_stream = self.graph.input_stream('source')

        # create overlay stream
        overlay_streams = []
        for (i, _) in enumerate(option['overlays']):
            self.inputs.append('overlay_' + str(i))
            overlay_streams.append(self.graph.input_stream('overlay_' + str(i)))

        # pre-processing for source layer
        info = option['source']
        output_stream = (
            source_stream.scale(info['width'], info['height'])
                .trim(start=info['start'], duration=info['duration'])
                .setpts('PTS-STARTPTS')
        )

        # overlay processing
        for (i, overlay_stream) in enumerate(overlay_streams):
            overlay_info = option['overlays'][i]

            # overlay layer pre-processing
            p_overlay_stream = (
                overlay_stream.scale(overlay_info['width'], overlay_info['height'])
                    .loop(loop=overlay_info['loop'], size=10000)
                    .setpts('PTS+%f/TB' % (overlay_info['start']))
            )

            # calculate overlay parameter
            x = 'if(between(t,%f,%f),%s,NAN)' % (overlay_info['start'],
                                                 overlay_info['start'] + overlay_info['duration'],
                                                 str(overlay_info['pox_x']))
            y = 'if(between(t,%f,%f),%s,NAN)' % (overlay_info['start'],
                                                 overlay_info['start'] + overlay_info['duration'],
                                                 str(overlay_info['pox_y']))
            if overlay_info['loop'] == -1:
                repeat_last = 0
                shortest = 1
            else:
                repeat_last = overlay_info['repeat_last']
                shortest = 1

            # do overlay
            output_stream = (
                output_stream.overlay(p_overlay_stream, x=x, y=y,
                                      repeatlast=repeat_last)
            )

        # finish creating graph
        self.output_streams = self.finish_create_graph([output_stream])
