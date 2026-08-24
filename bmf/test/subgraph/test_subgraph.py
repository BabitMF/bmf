import sys
import time
import unittest

sys.path.append("../../..")
sys.path.append("../../c_module_sdk/build/bin/lib")
import bmf
import os
if os.name == 'nt':
    # We redefine timeout_decorator on windows
    class timeout_decorator:

        @staticmethod
        def timeout(*args, **kwargs):
            return lambda f: f  # return a no-op decorator
else:
    import timeout_decorator

sys.path.append("../../test/")
from base_test.base_test_case import BaseTestCase
from base_test.media_info import MediaInfo


class OutputQueue(list):

    def put(self, packet):
        self.append(packet)


class EofTask:

    def __init__(self):
        self.output_queue = OutputQueue()
        self.timestamp = None

    def get_inputs(self):
        return {}

    def get_outputs(self):
        return {0: self.output_queue}

    def set_timestamp(self, timestamp):
        self.timestamp = timestamp


class EofGraph:

    def __init__(self):
        self.packet = bmf.Packet.generate_eof_packet()
        self.force_closed = False

    def poll_packet(self, stream):
        packet, self.packet = self.packet, None
        return packet

    def force_close(self):
        self.force_closed = True


class TestSubgraphEofHandling(unittest.TestCase):

    def test_process_closes_after_all_outputs_reach_eof(self):
        graph = EofGraph()
        task = EofTask()
        subgraph = object.__new__(bmf.SubGraph)
        subgraph.graph = graph
        subgraph.inputs = []
        subgraph.output_streams = [object()]
        subgraph.stream_done = {}
        subgraph.node_id_ = 0

        result = subgraph.process(task)

        self.assertEqual(result, bmf.ProcessResult.OK)
        self.assertEqual(task.timestamp, bmf.Timestamp.DONE)
        self.assertTrue(graph.force_closed)
        self.assertIsNone(subgraph.graph)
        self.assertEqual(len(task.output_queue), 2)
        for packet in task.output_queue:
            self.assertEqual(packet.get_timestamp(), bmf.Timestamp.EOF)


class TestSubgraph(BaseTestCase):

    @timeout_decorator.timeout(seconds=120)
    def test_subgraph(self):
        input_video_path = "../../files/big_bunny_10s_30fps.mp4"
        input_over_lay_image = "../../files/overlay.png"
        output_path = "./output.mp4"
        expect_result = '../subgraph/output.mp4|1080|1920|10.008|MOV,MP4,M4A,3GP,3G2,MJ2|1946098|2434569|h264|' \
                        '{"fps": "30.0662251656"}'
        self.remove_result_data(output_path)
        # create graph
        graph = bmf.graph()

        # decode video
        video = graph.decode({'input_path': input_video_path})

        # decoder overlay image
        overlay = graph.decode({'input_path': input_over_lay_image})

        # call sub graph and encoder
        (bmf.module([video['video'], overlay['video']],
                    'subgraph_module').encode(video['audio'], {
                        "output_path": output_path
                    }).run())
        self.check_video_diff(output_path, expect_result)


if __name__ == '__main__':
    unittest.main()
