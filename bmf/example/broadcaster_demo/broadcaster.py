import time
import json
from queue import Queue
import threading
import http_server

import bmf
from bmf import Log, LogLevel

import argparse

#input1 = "rtmp://localhost/live/zx"
#input2 = "rtmp://localhost/live/jyt"
#input3 = "rtmp://localhost/live/xx"
#input4 = "rtmp://localhost/live/xx1"

default_output = "rtmp://localhost/live/output"


MaxInputSources = 10

EosCallbackKey = 0

def get_source_alias(index):
    return "input_" + str(index)

#def process_eos_stream(param):
#    #p = json.loads(param.decode("utf-8"))
#    p = 1
#    Log.log(LogLevel.INFO, "param: ", param, " p: ", p)

# b.start()
# b.add_source(index, source_url)
# b.remove_source(index)
# b.set_volume(index, volume)
# b.set_layout(mode)
class Broadcaster:
    def __init__(self, rtmp_output = ''):
        self.rtmp_output = rtmp_output
        self.source_dict = dict()
        self.layout_alias = "layout"
        self.amix_alaias = "audiomix"
        self.streamhub_alias = "streamhub"

        # we know our rtmp source streams num is 2, and video is stream id 0, audio was stream id 1
        self.source_stream_num = 2
        self.schedule_cnt = 0
        self.current_max_schedule_id = 0
        self.streamhub_pads = dict()
        self.streamhub_stream_id = 0

        self.graph = bmf.graph()

    def process_eos_stream(self, param):
        p = json.loads(param.decode("utf-8"))
        Log.log(LogLevel.INFO, "param: ", param, " p: ", p)

        # TODO: delete pad info thread safe

        return bytes("OK","utf-8")

    def start(self):
        schedule_cnt = 1

        # main_stream = self.graph.decode(
        #    {"input_path": input1, "delay_init_input": 1, "alias": get_source_alias(1)}
        # )
        # self.source_dict[1] = input1
        # main_stream.node_.scheduler_ = schedule_cnt
        # schedule_cnt += 1

        wallclock = self.graph.module(
            "WallClock",
            option={"video_step": "1/25", "audio_step": "1024/44100"},
            module_path="",
            entry="wall_clock.WallClock",
            input_manager="immediate",
        )
        wallclock.node_.scheduler_ = schedule_cnt
        schedule_cnt += 1

        streamhub = bmf.module(
            [wallclock[0], wallclock[1]],
            "Streamhub",
            option={"alias": self.streamhub_alias},
            module_path="",
            entry="streamhub.Streamhub",
            input_manager="immediate",
        )
        self.streamhub_stream_id += 2
        streamhub.node_.scheduler_ = schedule_cnt
        streamhub.get_node().add_user_callback(0, self.process_eos_stream)
        schedule_cnt += 1

        output_video_stream = bmf.module(
            [streamhub[0]],
            "video_layout",
            option={
                "alias": "layout",
                "layout_mode": "gallery",
                "crop_mode": "",
                "layout_location": "",
                "interspace": 0,
                "main_stream_idx": 0,
                "width": 1280,
                "height": 720,
                "background_color": "#123456",
            },
            # module_path="/home/huheng.1989/source/bmf_vcloud_edit/video_layout",
            module_path="",
            entry="video_layout.video_layout",
            input_manager="immediate",
        )
        output_video_stream.node_.scheduler_ = schedule_cnt
        schedule_cnt += 1

        # audio
        output_audio_stream = bmf.module(
            [streamhub[1]],
            "Audiomix",
            option={"alias": self.amix_alaias},
            module_path="",
            entry="audiomix.Audiomix",
            input_manager="immediate",
        )
        output_audio_stream.node_.scheduler_ = schedule_cnt
        schedule_cnt += 1

        # video.run()
        bmf.encode(
            output_video_stream,
            output_audio_stream,
            {
                "video_params": {"g": "50", "preset": "veryfast", "g": 50, "bf": 0},
                # "audio_params": {"sample_rate": 44100, "codec": "aac"},
                "loglevel": "info",
                "output_path": self.rtmp_output,
                "format": "flv",
                "alias": "main_rtmp_output",
            },
        ).node_.scheduler_ = schedule_cnt

        self.schedule_cnt = schedule_cnt + MaxInputSources + 1
        self.current_max_schedule_id = schedule_cnt + 1

        self.graph.option_["scheduler_count"] = self.schedule_cnt
        self.graph.run_wo_block()

    def add_source(self, index, source_url):
        # volume is a float in range[0, 10], 1 means keep the origin volume
        if index < 0 or index >= MaxInputSources:
            Log.log(
                LogLevel.ERROR,
                "index: ",
                index,
                " out of range, max index: ",
                MaxInputSources - 1,
            )
            return
        if index in self.source_dict:
            Log.log(LogLevel.WARNING, "index: ", index, " was exist, use another index")
            return

        if source_url.index("rtmp") != 0:
            Log.log(
                LogLevel.ERROR,
                "broadcaster only support rtmp url, your source_url: ",
                source_url,
            )
            return

        self.source_dict[index] = source_url

        # add rtmp source
        update_source_graph = bmf.graph()
        input_alias = get_source_alias(index)
        video2 = update_source_graph.decode(
            {
                "input_path": source_url,
                "delay_init_input": 1,
                "alias": input_alias,
                # "video_params": {"extract_frames": {"fps": 25}},
            }
        )
        video2.node_.scheduler_ = self.current_max_schedule_id + index % MaxInputSources
        #ov = video2["video"].pass_through()
        #ov = video2["video"].scale(640,320)
        #video2.node_.scheduler_ = self.current_max_schedule_id + index % MaxInputSources
        output_config = {
            "alias": self.streamhub_alias,
            "streams": 2,
        }

        #oa = video2["audio"].pass_through()
        #ov.node_.scheduler_ = self.current_max_schedule_id + index % MaxInputSources

        update_source_graph.dynamic_add(video2, None, output_config)

        #update_source_graph.dynamic_add(oa, None, output_config)

        # update config
        self.streamhub_pads[self.streamhub_stream_id] = {
            "source_index": index,
            "type": "video",
            "is_live": True,
        }

        self.streamhub_pads[self.streamhub_stream_id + 1] = {
            "source_index": index,
            "type": "audio",
            "is_live": True,
        }

        Log.log(LogLevel.INFO, "after add, pad info: ", self.streamhub_pads)

        self.streamhub_stream_id += 2

        update_source_graph1 = bmf.graph()
        update_source_graph1.dynamic_reset(
            option={"alias": self.streamhub_alias, "pad_infos": self.streamhub_pads}
        )
        self.graph.update(update_source_graph1)

        graph_config_str = update_source_graph.graph_config_.dump()
        Log.log(LogLevel.INFO, "....update graph config str: ", graph_config_str)

        self.graph.update(update_source_graph)

    def get_all_stream_id(self, index):
        stream_id_list = []
        Log.log(LogLevel.INFO, "get all stream id, pad info: ", self.streamhub_pads)
        for (pad_stream_id, pad_info) in self.streamhub_pads.items():
            if pad_info["source_index"] == index:
                stream_id_list.append(pad_stream_id)
        return stream_id_list

    def remove_source(self, index):
        if index < 0 or index >= MaxInputSources:
            Log.log(
                LogLevel.ERROR,
                "index: ",
                index,
                " out of range, max index: ",
                MaxInputSources - 1,
            )
            return
        if index not in self.source_dict:
            Log.log(
                LogLevel.WARNING, "index: ", index, " was not exist, no need remove"
            )
            return

        # remove input source node and passthrough
        update_graph = bmf.graph()
        update_graph.dynamic_remove({"alias": get_source_alias(index)})
        self.graph.update(update_graph)

        self.source_dict.pop(index)

        stream_id_list = self.get_all_stream_id(index)
        for stream_id in stream_id_list:
            self.streamhub_pads.pop(stream_id)

    def get_stream_id(self, index, stream_type):
        # get stream_id
        stream_id = -1
        Log.log(LogLevel.INFO, "get stream id, pad info: ", self.streamhub_pads)
        for (pad_stream_id, pad_info) in self.streamhub_pads.items():
            if pad_info["source_index"] == index and pad_info["type"] == stream_type:
                stream_id = pad_stream_id
                break
        return stream_id

    def set_volume(self, index, volume):
        # volume is a float in range[0, 10], 1 means keep the origin volume
        if index < 0 or index >= MaxInputSources:
            Log.log(
                LogLevel.ERROR,
                "index: ",
                index,
                " out of range, max index: ",
                MaxInputSources - 1,
            )
            return

        stream_id = self.get_stream_id(index, "audio")
        if stream_id == -1:
            Log.log(LogLevel.WARNING, "index: ", index, " has no audio steam")
            return

        if volume < 0 or volume > 10.0:
            Log.log(LogLevel.ERROR, "volume: ", volume, " out of range, use value in 0-10.0")
            return

        config = dict()
        config[stream_id] = volume
        config["alias"] = self.amix_alaias

        update_graph = bmf.graph()
        update_graph.dynamic_reset(config)
        self.graph.update(update_graph)

    def set_layout(self, layout):
        # if index < 0 or index >= MaxInputSources:
        #    Log.log(
        #        LogLevel.ERROR,
        #        "index: ",
        #        index,
        #        " out of range, max index: ",
        #        MaxInputSources - 1,
        #    )
        #    return
        # stream_id = self.get_stream_id(index, "video")
        # if stream_id == -1:
        #    Log.log(LogLevel.WARNING, "index: ", index, " has no video steam")
        #    return

        layout["alias"] = self.layout_alias
        update_graph = bmf.graph()
        update_graph.dynamic_reset(layout)
        self.graph.update(update_graph)

    def do_inspect(self, path):
        return json.dumps(self.source_dict)

    def show_sources(self):
        Log.log(LogLevel.INFO, " sources: ", self.source_dict)


def process_rpc_request(broadcaster, rpc_queue, send_queue):
    Log.log(LogLevel.INFO, "process rpc request..........................")
    while True:
        item = rpc_queue.get()
        try:
            rpc = json.loads(item)
            Log.log(LogLevel.INFO, "item:", item, "rpc: ", rpc)

            if rpc.get("method", None) is None:
                Log.log(LogLevel.WARNING, "invalid rpc request")
                continue

            method = rpc["method"]
            if method == "add_source":
                broadcaster.add_source(rpc["index"], rpc["input_path"])
            elif method == "remove_source":
                broadcaster.remove_source(rpc["index"])
            elif method == "set_volume":
                broadcaster.set_volume(rpc["index"], rpc["volume"])
            elif method == "set_layout":
                broadcaster.set_layout(rpc["layout"])
            elif method == "inspect":
                data = broadcaster.do_inspect(rpc["path"])
                send_queue.put(data)
            else:
                Log.log(LogLevel.WARNING, "invalid rpc method: ", rpc["method"])

        except Exception as e:
            print("type error: " + str(e))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='use an rtmp addr as broadcaster output')
    parser.add_argument('--output', type=str, help='rtmp output addr', default=default_output)
    args = parser.parse_args()

    print("rtmp output addr: ", args.output)

    broadcaster = Broadcaster(args.output)
    broadcaster.start()

    rpc_queue = Queue(1)
    send_queue = Queue(1)
    t = threading.Thread(target=http_server.run, args=(rpc_queue,send_queue,))
    t.start()
    process_rpc_request(broadcaster, rpc_queue, send_queue)

