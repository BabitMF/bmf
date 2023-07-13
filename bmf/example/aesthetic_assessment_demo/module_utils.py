#!/usr/bin/env python
# -*- coding: utf-8 -*
import bmf

from bmf import VideoFrame
from bmf.lib._bmf.sdk import ffmpeg
import bmf.hml.hmp as mp


def generate_out_packets(packet, np_arr, out_fmt):
    # video_frame = bmf.VideoFrame.from_ndarray(np_arr, format=out_fmt)
    rgbformat = mp.PixelInfo(mp.kPF_RGB24)
    image = mp.Frame(mp.from_numpy(np_arr), rgbformat)
    video_frame = VideoFrame(image)

    video_frame.pts = packet.get(VideoFrame).pts
    video_frame.time_base = packet.get(VideoFrame).time_base

    pkt = bmf.Packet(video_frame)
    pkt.timestamp = packet.timestamp
    return pkt


class SyncModule(bmf.Module):
    def __init__(self, node=None, nb_in=1, in_fmt="yuv420p", out_fmt="yuv420p"):
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

        input_queue = task.get_inputs()[0]
        # output_queue = task.get_outputs()[0]

        while not input_queue.empty():
            pkt = input_queue.get()
            pkt_timestamp = pkt.timestamp

            if pkt_timestamp == bmf.Timestamp.EOF:
                self._eof = True
                for _ in range(self._margin_num):
                    self._in_packets.append(self._in_packets[-1])
                    self._frames.append(self._frames[-1])
                self._consume()

                # output_queue.put(bmf.Packet.generate_eof_packet())
                task.set_timestamp(bmf.Timestamp.DONE)
                return bmf.ProcessResult.OK

            pkt_data = pkt.get(VideoFrame)
            if pkt_data is not None:
                self._in_packets.append(pkt)
                # self._frames.append(pkt.get(VideoFrame).to_ndarray(format=self._in_fmt))

                self._frames.append(
                    ffmpeg.reformat(pkt.get(VideoFrame), self._in_fmt)
                    .frame()
                    .plane(0)
                    .numpy()
                )

            # padding first frame.
            if len(self._in_packets) == 1:
                for _ in range(self._margin_num):
                    self._in_packets.append(self._in_packets[0])
                    self._frames.append(self._frames[0])

        self._consume()

        return bmf.ProcessResult.OK

    def _consume(self, output_queue=None):
        while len(self._in_packets) >= self._in_frame_num:
            out_frame = self.core_process(self._frames[: self._in_frame_num])
            out_packet = generate_out_packets(
                self._in_packets[self._out_frame_index], out_frame, self._out_fmt
            )
            # output_queue.put(out_packet)
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
