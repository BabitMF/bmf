from bmf import SubGraph

'''
Option example:
    option = {
        'Type': 'image',
        'Source': '../files/blue.png',
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
        'Rotate': 0,
        "Filters": {
            "Contrast": 110,
            "Brightness": 62,
            "Saturate": 121,
            "Opacity": 66,
            "Blur": 28
        },
        'ExtraFilters': {
            'Type': 'AAA',
            'Para': 'BBB',
        }
    }
'''


class image_preprocess(SubGraph):
    def create_graph(self, option=None):
        # create source stream for subgraph
        self.inputs.append('source_image')
        stream = self.graph.input_stream('source_image')

        # do image scale
        position = option.get('Position')
        if position:
            stream = stream.scale(position['Width'], position['Height'])

        # do image crop
        crop = option.get('Crop')
        if crop:
            stream = stream.ff_filter('crop', crop['Width'], crop['Height'], crop['PosX'], crop['PosY'])

        # do basic filters
        filters = option.get('Filters')
        if filters:
            stream = stream.ff_filter('eq', contrast=filters['Contrast'],
                                      brightness=filters['Brightness'], saturation=filters['Saturate'])

        # finish creating graph
        self.output_streams = self.finish_create_graph([stream])
