import bmf
from bmf import SubGraph


class simple_audio_concat(SubGraph):

    def create_graph(self, option=None):
        streamName1 = "simple_concat01"
        self.inputs.append(streamName1)
        video1 = self.graph.input_stream(streamName1)
        streamName2 = "simple_concat02"
        self.inputs.append(streamName2)
        video2 = self.graph.input_stream(streamName2)

        video = bmf.concat(video1, video2, v=0, a=1)

        # run graph, get output streams
        self.output_streams = self.finish_create_graph([video])
