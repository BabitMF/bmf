import numpy as np
from bmf import (
    Module,
    Log,
    LogLevel,
    InputType,
    ProcessResult,
    Packet,
    AudioFrame,
    Timestamp,
    scale_av_pts,
    av_time_base,
    BmfCallBackType,
)

from bmf.lib._bmf.sdk import Rational
from bmf.lib._bmf import sdk

import bmf.hml.hmp as mp


nb_samples = 1024
channels = 2
bytes_per_sample = 4
sample_rate = 44100

# assume every upstream node has two input streams, 0 is video, 1 is audio
# make timestamp of all streams in same timeline
class Audiomix(Module):
    def __init__(self, node, option=None):
        self.node_ = node
        self.option = option
        # we only support audio frame: 44100 lc-aac, fltp, stereo 1024 samples
        self.timebase = Rational(1, sample_rate)
        self.volume_table = dict()

    def reset(self):
        Log.log_node(LogLevel.DEBUG, self.node_, " is doing reset")

    def dynamic_reset(self, opt_reset=None):
        Log.log_node(
            LogLevel.INFO,
            self.node_,
            "opt_reset type:",
            type(opt_reset),
            "opt_reset: ",
            opt_reset,
        )
        if opt_reset is None:
            return
        if self.option is None:
            self.option = dict()
        for (para, value) in opt_reset.items():
            self.option[para] = value
        Log.log_node(
            LogLevel.INFO,
            self.node_,
            "opt_reset:",
            opt_reset,
            "self.option: ",
            self.option,
        )

    def get_volume(self, stream_id):
        if self.option is None or self.option.get(str(stream_id), None) is None:
            return 1
        return self.option[str(stream_id)]

    def do_mix(self, audio_list):
        # mixed_frame = AudioFrame(1024, sdk.kLAYOUT_STEREO, True, dtype=mp.float32)
        # mixed_frame.sample_rate = sample_rate
        # return mixed_frame
        # planes = mp.to_numpy(mixed_frame.planes)
        # mixed_array = np.array(planes)
        mixed_array = np.zeros([channels, nb_samples], dtype=np.float32)
        Log.log(LogLevel.DEBUG, "np shape: ", mixed_array.shape)
        for (index, pkt) in audio_list:
            if index == -1:
                continue

            frame = pkt.get(AudioFrame)
            if (
                frame.sample_rate != sample_rate
                or frame.nsamples != nb_samples
                or frame.layout != sdk.kLAYOUT_STEREO
            ):
                Log.log(
                    LogLevel.DEBUG,
                    "frame property not match!!!",
                    " nsamples: ",
                    frame.nsamples,
                    "layout: ",
                    frame.layout,
                )
                continue

            frame_planes = mp.to_numpy(frame.planes)
            frame_array = np.array(frame_planes)
            Log.log(
                LogLevel.DEBUG,
                "frame array shape: ",
                frame_array.shape,
                "dtype: ",
                frame_array.dtype,
            )
            mixed_array += frame_array * self.get_volume(index)
            Log.log(
                LogLevel.DEBUG,
                "frame nsamples: ",
                frame.nsamples,
                "layout: ",
                frame.layout,
                "dtype: ",
                frame.dtype,
                "mix_array shape: ",
                mixed_array.shape,
                "mix_array dtype: ",
                mixed_array.dtype,
            )

        # data = mp.from_numpy(mixed_array.tolist())
        data = []
        for m in mixed_array:
            data.append(mp.from_numpy(m))
        Log.log(LogLevel.DEBUG, "..........data: ", data.__class__)
        mixed_frame = AudioFrame(data, sdk.kLAYOUT_STEREO)
        mixed_frame.sample_rate = sample_rate
        Log.log(
            LogLevel.DEBUG,
            "mixed frame info, frame sample_rate:",
            mixed_frame.sample_rate,
            "samples:",
            mixed_frame.nsamples,
            "layout:",
            mixed_frame.layout,
            "dtype:",
            mixed_frame.dtype,
        )
        return mixed_frame

    def process(self, task):
        output_queue = task.get_outputs().get(0, None)
        for (input_id, input_packets) in task.get_inputs().items():
            Log.log(LogLevel.DEBUG, "audiomix get input stream id:", input_id)
            while not input_packets.empty():
                timestamp = 0
                bmf_pkt = input_packets.get()
                frame_list = bmf_pkt.get(list)
                if frame_list:
                    timestamp = frame_list[0][1]

                Log.log(
                    LogLevel.DEBUG,
                    "audiomix do mix, framelist len",
                    len(frame_list),
                    "timestamp: ",
                    timestamp,
                )
                audio_frame = self.do_mix(frame_list)
                audio_frame.time_base = self.timebase
                audio_frame.pts = scale_av_pts(
                    timestamp,
                    av_time_base,
                    float(self.timebase.num) / self.timebase.den,
                )
                # Log.log(
                #    LogLevel.DEBUG,
                #    "audio mix output frame, sample_rate:",
                #    audio_frame.sample_rate,
                #    "samples:",
                #    audio_frame.samples,
                #    "layout:",
                #    audio_frame.layout,
                #    "format:",
                #    audio_frame.format,
                #    "timestamp:",
                #    timestamp,
                # )
                output_pkt = Packet(audio_frame)
                output_pkt.timestamp = timestamp

                if output_queue is not None:
                    output_queue.put(output_pkt)

        return ProcessResult.OK
