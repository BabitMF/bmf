import os
import sys
import bmf

cmd_options = {
    'decoder': ['i', 'v', 'a', 'map_v', 'map_a', 'start_time', 'end_time', 'fps', 'video_time_base', 'skip_frame',
                'video_codec', 'audio_codec', 'dec_params', 'decryption_key', 'autorotate'],
    'encoder': ['map', 'o', 'fmt'],
    'video_s': ['codec', 'crf', 'preset', 'width', 'height', 'x264-params', 'pix_fmt', 'threads', 'in_time_base',
                'max_fr', 'qscale', 'vtag'],
    'audio_s': ['codec', 'bit_rate', 'sample_rate', 'channels', 'qscale', 'atag'],
    'filter': ['name', 'para'],
    'module': ['name', 'option', 'module_path', 'entry']
}

graph = bmf.graph({"dump_graph": 1, "graph_name": "bmf_cmd"})
bmf_stream = {}


def parse_decoder_option(param_list):
    video_stream = None
    audio_stream = None
    curr_para = None
    decoder_option = {}

    # record decoder option
    while len(param_list) > 0:
        para = param_list.pop(0)
        if len(para) > 1 and para[0] == "-":
            if para[1:] not in cmd_options["decoder"]:
                print("incorrect decoder option")
                os._exit(1)
            curr_para = para[1:]
            continue
        if curr_para is None:
            print("incorrect decoder option")
            os._exit(1)
        if curr_para == "i":
            decoder_option["input_path"] = para
        elif curr_para == "v":
            video_stream = para[1:-1]
        elif curr_para == "a":
            audio_stream = para[1:-1]
        else:
            try:
                para = int(para)
            except:
                pass
            decoder_option[curr_para] = para

    # check input path
    if 'input_path' not in decoder_option.keys():
        print("decoder has no input path")
        os._exit(1)

    # check output streams for decoder
    if video_stream is None and audio_stream is None:
        print("decoder has no output stream")
        os._exit(1)

    # create decoder node
    decoder_node = graph.decode(decoder_option)
    global bmf_stream
    if video_stream is not None:
        bmf_stream[video_stream] = decoder_node['video']
    if audio_stream is not None:
        bmf_stream[audio_stream] = decoder_node['audio']


def parse_video_option(video_params_list):
    curr_para = None
    video_params = {}

    # record encoder video params one by one
    while len(video_params_list) > 0:
        para = video_params_list.pop(0)
        if len(para) > 1 and para[0] == "-":
            if para[1:] not in cmd_options["video_s"]:
                print("incorrect video stream option")
                os._exit(1)
            curr_para = para[1:]
            continue
        if curr_para is None:
            print("incorrect video stream option")
            os._exit(1)

        # value must be integer
        if curr_para == "width" or curr_para == "height":
            para = int(para)

        # record video param
        video_params[curr_para] = para

    # return encoder video params
    return video_params


def parse_audio_option(audio_params_list):
    curr_para = None
    audio_params = {}

    # recoder encoder audio params one by one
    while len(audio_params_list) > 0:
        para = audio_params_list.pop(0)
        if len(para) > 1 and para[0] == "-":
            if para[1:] not in cmd_options["audio_s"]:
                print("incorrect audio stream option")
                os._exit(1)
            curr_para = para[1:]
            continue
        if curr_para is None:
            print("incorrect audio stream option")
            os._exit(1)

        # must be integer
        if curr_para == "channels":
            para = int(para)

        # record audio param
        audio_params[curr_para] = para

    # return encoder audio params
    return audio_params


def parse_encoder_option(opt_list):
    index = 0
    output_path = None
    format = None
    v_stream = None
    a_stream = None
    video_params = {}
    audio_params = {}

    # record encoder params
    while index < len(opt_list):
        opt = opt_list[index]
        index += 1
        if opt == "-map":
            end_index = index
            tag = None
            while end_index < len(opt_list):
                if opt_list[end_index] == "-map" or opt_list[end_index] == "-o" or opt_list[end_index] == "-fmt":
                    break
                if tag is None and opt_list[end_index][:2] == "[v":
                    tag = "video_stream"
                    v_stream_tag = opt_list[end_index][1:-1]
                    if v_stream_tag not in bmf_stream.keys():
                        print("incorrect video stream tag")
                        os._exit(1)
                    v_stream = bmf_stream[v_stream_tag]
                elif tag is None and opt_list[end_index][:2] == "[a":
                    tag = "audio_stream"
                    a_stream_tag = opt_list[end_index][1:-1]
                    if a_stream_tag not in bmf_stream.keys():
                        print("incorrect audio stream tag")
                        os._exit(1)
                    a_stream = bmf_stream[a_stream_tag]
                end_index += 1
            if end_index - index < 1:
                print("lost stream tag after param -map")
                os._exit(1)
            if tag == "video_stream":
                video_params = parse_video_option(opt_list[index + 1: end_index])
            elif tag == "audio_stream":
                audio_params = parse_audio_option(opt_list[index + 1: end_index])
            index = end_index
        elif opt == "-o":
            output_path = opt_list[index]
            index += 1
        elif opt == "-fmt":
            format = opt_list[index]
            index += 1
        else:
            print("incorrect option")
            os._exit(1)

    # create encoder option
    encoder_option = {}
    encoder_option["output_path"] = output_path

    if format is not None:
        encoder_option["format"] = format
    if len(video_params) > 0:
        encoder_option["video_params"] = video_params
    if len(audio_params) > 0:
        encoder_option["audio_params"] = audio_params

    # create encoder node
    bmf.encode(v_stream, a_stream, encoder_option)


def parse_single_filter_info(info):
    src_list = info["src"]
    dst_list = info["dst"]

    # ensure src stream is already available
    for src in src_list:
        if src not in bmf_stream.keys():
            print("stream ", src, " not exits")
            os._exit(1)

    # record filter para
    filter_option = {}
    curr_para = None
    while len(info["info_list"]) > 0:
        para = info["info_list"].pop(0)
        if len(para) > 1 and para[0] == "-":
            if para[1:] not in cmd_options["filter"]:
                print("incorrect filter option")
                os._exit(1)
            curr_para = para[1:]
            continue
        elif curr_para is None:
            print("incorrect filter option")
            os._exit(1)
        filter_option[curr_para] = para

    # get filter input stream
    filter_input_streams = []
    for src in src_list:
        filter_input_streams.append(bmf_stream[src])

    # create filter output stream
    filter_output_streams = bmf.module(filter_input_streams, "c_ffmpeg_filter", filter_option)
    for i in range(len(dst_list)):
        bmf_stream[dst_list[i]] = filter_output_streams[i]


def parse_filter_option(filter_options):
    filter_list = []

    # collect filters
    filter_option_str = " ".join(filter_options)
    filter_option_list = filter_option_str.split(";")
    for filter_desc in filter_option_list:
        src_tag = []
        dst_tag = []

        # find filter src streams
        while True:
            filter_desc = filter_desc.strip()
            src_tag_begin_idx = filter_desc.find("[")
            if (src_tag_begin_idx > 0):
                break
            src_tag_end_idx = filter_desc.find("]")
            src_tag.append(filter_desc[1: src_tag_end_idx])
            filter_desc = filter_desc[src_tag_end_idx + 1:]

        if len(src_tag) == 0:
            print("no input streams for filter")
            os._exit(1)

        if filter_desc.find("[") < 0:
            print("no output streams for filter")
            os._exit(1)

        # find filter option
        info_list = filter_desc[:filter_desc.find("[")].strip().split(" ")

        # find filter dst streams
        while True:
            dst_tag_begin_idx = filter_desc.find("[")
            if (dst_tag_begin_idx < 0):
                break
            dst_tag_end_idx = filter_desc.find("]")
            dst_tag.append(filter_desc[dst_tag_begin_idx + 1: dst_tag_end_idx])
            filter_desc = filter_desc[dst_tag_end_idx + 1:]

        # collect filter info
        filter_list.append({
            "src": src_tag,
            "dst": dst_tag,
            "info_list": info_list
        })

    # parse and create every single filter
    for filter_info in filter_list:
        parse_single_filter_info(filter_info)


def parse_module_option(module_options):
    option_desc = " ".join(module_options)
    curr_para = None
    name = None
    option = {}
    module_path = ""
    entry = ""
    src_tag = []
    dst_tag = []

    # find src streams
    while True:
        option_desc = option_desc.strip()
        src_tag_begin_idx = option_desc.find("[")
        if (src_tag_begin_idx > 0):
            break
        src_tag_end_idx = option_desc.find("]")
        src_tag.append(option_desc[1: src_tag_end_idx])
        option_desc = option_desc[src_tag_end_idx + 1:]

    if len(src_tag) == 0:
        print("no module input streams")
        os._exit(1)

    if option_desc.find("[") < 0:
        print("no module output streams")
        os._exit(1)

    # find module option
    param_list = option_desc[:option_desc.find("[")].strip().split(" ")

    # find dst streams
    while True:
        dst_tag_begin_idx = option_desc.find("[")
        if (dst_tag_begin_idx < 0):
            break
        dst_tag_end_idx = option_desc.find("]")
        dst_tag.append(option_desc[dst_tag_begin_idx + 1: dst_tag_end_idx])
        option_desc = option_desc[dst_tag_end_idx + 1:]

    # record module option
    while len(param_list) > 0:
        para = param_list.pop(0)
        if len(para) > 1 and para[0] == "-":
            if para[1:] not in cmd_options["module"]:
                print("incorrect module option")
                os._exit(1)
            curr_para = para[1:]
            continue
        if curr_para is None:
            print("incorrect module option")
            os._exit(1)
        if curr_para == "name":
            name = para
        elif curr_para == "option":
            option = para
        elif curr_para == "module_path":
            module_path = para
        elif curr_para == "entry":
            entry = para

    # get input stream
    module_input_streams = []
    for src in src_tag:
        module_input_streams.append(bmf_stream[src])

    # turn option to dict
    import ast
    option_dict = ast.literal_eval(option)

    # create output stream
    module_output_streams = bmf.module(module_input_streams, name, option_dict, module_path, entry)
    for i in range(len(dst_tag)):
        bmf_stream[dst_tag[i]] = module_output_streams[i]


def parse_option(name, param_list):
    if name == "decoder":
        parse_decoder_option(param_list)

    elif name == "encoder":
        parse_encoder_option(param_list)

    elif name == "filter":
        parse_filter_option(param_list)

    elif name == "module":
        parse_module_option(param_list)

    else:
        print("option is not correct")
        os._exit(1)


def parse_options(opt_list):
    index = 1
    while index < len(opt_list):
        opt = opt_list[index]
        index += 1
        if len(opt) > 2 and opt[:2] == "--":
            if opt[2:] not in cmd_options.keys():
                print("incorrect option")
                os._exit(1)
            end_index = index
            while end_index < len(opt_list):
                if len(opt_list[end_index]) > 2 and opt_list[end_index][:2] == "--":
                    break
                end_index += 1
            if end_index - index == 0:
                print("incorrect option")
                os._exit(1)
            parse_option(opt[2:], opt_list[index: end_index])
            index = end_index
        else:
            print("incorrect option")
            os._exit(1)


if __name__ == '__main__':
    bmf.Log.set_log_level(bmf.LogLevel.DEBUG)

    # parse cmd options and build graph
    parse_options(sys.argv)

    # run graph
    graph.run()

