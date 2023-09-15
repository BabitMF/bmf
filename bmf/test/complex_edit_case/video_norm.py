from bmf import SubGraph


class video_norm(SubGraph):

    def create_graph(self, option=None):
        stream_name = "video_norm"
        self.inputs.append(stream_name)
        video = self.graph.input_stream(stream_name)

        video = video.fps(20)
        video = video.scale(-2, 720)
        video = video.setsar(r="1/1")
        video = video.pad(w=1280,
                          h=720,
                          x="(ow-iw)/2",
                          y="(oh-ih)/2",
                          color="black")

        # run graph, get output streams
        self.output_streams = self.finish_create_graph([video])
