from bmf import SubGraph


class subgraph_module(SubGraph):
    def create_graph(self, option=None):
        # use self.graph to build a processing graph
        # put name of input streams into self.inputs

        # input stream name, used to fill packet in
        self.inputs.append('video')
        self.inputs.append('overlay')

        # create input streams
        in_stream_0 = self.graph.input_stream('video')
        in_stream_1 = self.graph.input_stream('overlay')

        # output stream
        output_stream = (
            in_stream_0.vflip()
                .overlay(in_stream_1)
        )

        # finish creating graph
        self.output_streams = self.finish_create_graph([output_stream])
