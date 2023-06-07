import time
import json
from bmf import (
    Module,
    Log,
    LogLevel,
    InputType,
    ProcessResult,
    Packet,
    VideoFrame,
    AudioFrame,
    Timestamp,
    scale_av_pts,
    av_time_base,
    BmfCallBackType,
)

from jitter_buffer import JitterBuffer
from fractions import Fraction


def output_stream_type(stream_id):
    if stream_id % 2 == 0:
        return "video"
    else:
        return "audio"


def input_stream_type(stream_id, option):
    stream_id = str(stream_id)
    if (
        option is None
        or option.get("pad_infos", None) is None
        or option["pad_infos"].get(stream_id, None) is None
    ):
        return ""
    return option["pad_infos"][stream_id]["type"]


# assume every upstream node has two input streams, 0 is video, 1 is audio
# make timestamp of all streams in same timeline
# streamhub has two outputstream, 0 is video packet list stream, 1 is audio packet list stream
class Streamhub(Module):
    def __init__(self, node, option=None):
        self.node_ = node
        self.option = option
        # input_id:jitterbuffer
        self.buffer_dict = dict()

        # {0: 10}, 0 is stream id, 10 is offset which means timestamp of newtimeline equals timestamp of old timeline add 10
        self.timestamp_map = dict()
        self.source_stream_num = 2

        self.start_time_utc = None
        self.last_audio_timestamp = 0
        self.last_video_timestamp = 0

        Log.log_node(LogLevel.INFO, self.node_, "streamhub node init")

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

    def get_relevant_stream_id(self, stream_id):
        stream_id = str(stream_id)
        if (
            self.option is None
            or self.option.get("pad_infos", None) is None
            or self.option["pad_infos"].get(stream_id, None) is None
        ):
            return list()

        source_index = self.option["pad_infos"][stream_id]["source_index"]

        relevant_list = []
        for (relevant_stream_id, pad_info) in self.option["pad_infos"].items():
            if (
                relevant_stream_id != stream_id
                and pad_info["source_index"] == source_index
            ):
                relevant_list.append(int(relevant_stream_id))
        return relevant_list

    def relevant_stream_offset(self, stream_id, offset):
        relevant_list = self.get_relevant_stream_id(stream_id)
        Log.log_node(
            LogLevel.INFO,
            self.node_,
            "stream_id: ",
            stream_id,
            "relevant stream offset: ",
            offset,
            " relevant_stream_id: ",
            relevant_list,
        )
        for relevant_stream_id in relevant_list:
            if relevant_stream_id in self.buffer_dict:
                buffer = self.buffer_dict[relevant_stream_id]
                old_offset = buffer.get_offset()
                Log.log_node(
                    LogLevel.INFO,
                    self.node_,
                    "relevant stream old offset: ",
                    old_offset,
                    " offset: ",
                    offset,
                )
                if old_offset is None:
                    old_offset = 0
                if old_offset == offset:
                    return
                buffer.add_timestamp_offset(offset - old_offset)
                buffer.set_offset(offset)

    def cache_pkts(self, task):
        for (input_id, input_packets) in task.get_inputs().items():
            # input id 0, 1 is wall clock
            if input_id < 2:
                continue
            if input_packets.empty():
                Log.log_node(
                    LogLevel.DEBUG,
                    self.node_,
                    "input_id:",
                    input_id,
                    " empty...................",
                )
            while not input_packets.empty():
                pkt = input_packets.get()

                ist = input_stream_type(input_id, self.option)

                if input_id not in self.buffer_dict and (
                    ist == "video" or ist == "audio"
                ):
                    Log.log_node(
                        LogLevel.INFO,
                        self.node_,
                        "create jitterbuffer, input_id: ",
                        input_id,
                        "type: ",
                        ist,
                    )
                    self.buffer_dict[input_id] = JitterBuffer(
                        input_id,
                        input_stream_type(input_id, self.option) == "audio",
                        self.relevant_stream_offset,
                    )
                elif input_id not in self.buffer_dict:
                    Log.log_node(
                        LogLevel.DEBUG,
                        self.node_,
                        "cannot create jitterbuffer, input_id: ",
                        input_id,
                        "type: ",
                        ist,
                    )
                    continue

                buffer = self.buffer_dict[input_id]
                if pkt.timestamp == Timestamp.EOF:
                    Log.log_node(LogLevel.INFO, task.get_node(), "Receive EOF")
                    buffer.set_eof_get()
                elif pkt.timestamp == Timestamp.DYN_EOS:
                    Log.log_node(LogLevel.INFO, task.get_node(), "Receive DYN_EOS")
                    buffer.set_eof_get()
                elif pkt.timestamp == Timestamp.EOS:
                    Log.log_node(LogLevel.INFO, task.get_node(), "Receive EOS")
                    buffer.set_eof_get()

                elif pkt.timestamp != Timestamp.UNSET:
                    Log.log_node(
                        LogLevel.DEBUG,
                        self.node_,
                        "cache pkt, input_id: ",
                        input_id,
                        "timestamp: ",
                        pkt.timestamp,
                    )
                    # cache all pkt
                    buffer.push_packet(pkt)

    def process(self, task):
        # debug print id
        for (input_id, _) in task.get_inputs().items():
            Log.log_node(
                LogLevel.DEBUG,
                task.get_node(),
                "streamhub task input id: ",
                input_id,
            )

        self.cache_pkts(task)

        # output packet
        # list: [(streamid, pkt), (streamid, pkt)...]
        for (index, input_packets) in task.get_inputs().items():
            # input id 0, 1 is wall clock, 0, 1 is also output index
            if index >= 2:
                continue

            output_packets = task.get_outputs().get(index, None)
            while not input_packets.empty():
                pts_pkt = input_packets.get()
                pts = pts_pkt.timestamp
                out_pkts = []
                out_pkts.append((-1, pts))
                Log.log(LogLevel.DEBUG, "hub output index: ", index, "pts: ", pts)
                for (stream_id, buffer) in self.buffer_dict.items():
                    if output_stream_type(index) != input_stream_type(
                        stream_id, self.option
                    ):
                        continue
                    pkt = buffer.get_packet(pts)
                    if pkt is not None:
                        out_pkts.append((stream_id, pkt))
                        Log.log(
                            LogLevel.DEBUG,
                            "hub output index: ",
                            index,
                            "stream id: ",
                            stream_id,
                            "pkt pts: ",
                            pkt.timestamp,
                        )

                bmf_pkt = Packet(out_pkts)
                if output_packets is not None:
                    output_packets.put(bmf_pkt)

        buffer_eof_list = []
        for (stream_id, buffer) in self.buffer_dict.items():
            if buffer.is_empty() and buffer.is_get_eof_pkt():
                buffer_eof_list.append(stream_id)
                if self.callback_ is not None:
                    message = {"stream_id": stream_id}
                    self.callback_(0, bytes(json.dumps(message), "utf-8"))

        for stream_id in buffer_eof_list:
            Log.log(LogLevel.INFO, stream_id, " eof")
            self.buffer_dict.pop(stream_id)

            # delete eos pad info
            if (
                self.option is not None
                and self.option.get("pad_infos", None) is not None
            ):
                self.option["pad_infos"].pop(str(stream_id))

        return ProcessResult.OK
