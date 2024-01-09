import sys
if sys.version_info.major == 2:
    from Queue import *
else:
    from queue import *
import bmf

#from module_utils.util import generate_out_packets


def generate_out_packets(packet, np_arr, out_fmt):
    video_frame = bmf.VideoFrame.from_ndarray(np_arr, format=out_fmt)
    video_frame.pts = packet.get_data().pts
    video_frame.time_base = packet.get_data().time_base

    pkt = bmf.Packet()
    pkt.set_timestamp(packet.get_timestamp())
    pkt.set_data(video_frame)
    return pkt


class SyncModule(bmf.Module):
    def __init__(self, node=None, nb_in=1, in_fmt='yuv420p', out_fmt='yuv420p'):
        """
        nb_in: the number of frames for core_process function
        in_fmt: the pixel format of frames for core_process function
        out_fmt: the pixel format of frame returned by core_process function
        """
        self._node = node

        self._margin_num = (nb_in - 1) // 2
        self._out_frame_index = self._margin_num
        self._in_frame_num = nb_in

        self._in_fmt = in_fmt
        self._out_fmt = out_fmt

        self._in_packets = []
        self._frames = []
        self._eof = False

    def process(self, task):
        print(task.get_inputs().items(),'####',task.get_outputs().items())
        input_queue = task.get_inputs()[0]
        output_queue = task.get_outputs()[0]

        while not input_queue.empty():
            pkt = input_queue.get()
            pkt_timestamp = pkt.get_timestamp()
            pkt_data = pkt.get_data()
            print('##',pkt_data)


            if pkt_timestamp == bmf.Timestamp.EOF:
                self._eof = True
            if pkt_data is not None:
                self._in_packets.append(pkt)
                self._frames.append(pkt.get_data().to_ndarray(format=self._in_fmt))

            # padding first frame.
            if len(self._in_packets) == 1:
                for _ in range(self._margin_num):
                    self._in_packets.append(self._in_packets[0])
                    self._frames.append(self._frames[0])

        if self._eof:
            #print(self._in_packets, self._frames)
            # padding last frame.
            for _ in range(self._margin_num):
                self._in_packets.append(self._in_packets[-1])
                self._frames.append(self._frames[-1])
            self._consume(output_queue)

            output_queue.put(bmf.Packet.generate_eof_packet())
            task.set_timestamp(bmf.Timestamp.DONE)

        return bmf.ProcessResult.OK

    def _consume(self, output_queue):
        while len(self._in_packets) >= self._in_frame_num:
            out_frame = self.core_process(self._frames[:self._in_frame_num])
            out_packet = generate_out_packets(self._in_packets[self._out_frame_index], out_frame, self._out_fmt)
            output_queue.put(out_packet)
            self._in_packets.pop(0)
            self._frames.pop(0)

    def core_process(self, frames):
        """
        user defined, process frames to output one frame, pass through by default
        frames: input frames, list format
        """
        return frames[0]

    def clean(self):
        pass

    def close(self):
        self.clean()

    def reset(self):
        self._eof = False








