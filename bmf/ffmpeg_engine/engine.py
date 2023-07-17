import os


# ffmpeg need to be in PATH or the sys enviroment of FFMPEG_BIN_PATH need to be set.
class FFmpegEngine(object):

    def escaping_param(self, param):
        escap_char_list = "\\[],;"
        escaping_param_result = ""
        for char in param:
            if char in escap_char_list:
                escaping_param_result = escaping_param_result + "\\" + char
            else:
                escaping_param_result = escaping_param_result + char
        return escaping_param_result

    def get_decoder_command(self, node):
        decoder_command = ""
        if "start_time" in node.option.keys():
            decoder_command = decoder_command + "-ss " + str(
                node.option["start_time"]) + " "
        if "dec_params" in node.option.keys():
            for key, value in node.option["dec_params"].items():
                decoder_command = decoder_command + "-" + key + " " + str(
                    value) + " "

        if "decryption_key" in node.option.keys():
            decoder_command = decoder_command + " -decryption_key " + node.option[
                "decryption_key"] + " "
        output_video_flag = False
        output_audio_flag = False
        for output_stream in node.get_output_stream_names():
            if output_stream.split(":")[0] == "video":
                self.decoder_input_streams_[output_stream.split(
                    ":")[1]] = "[{}:v] ".format(self.decoder_index_)
                output_video_flag = True
            elif output_stream.split(":")[0] == "audio":
                output_audio_flag = True
                self.decoder_input_streams_[output_stream.split(
                    ":")[1]] = "[{}:a] ".format(self.decoder_index_)

        if not output_audio_flag:
            decoder_command = decoder_command + "-an "
        if not output_video_flag:
            decoder_command = decoder_command + "-vn "

        decoder_command = decoder_command + "-i " + node.option[
            "input_path"] + " "
        self.decoder_index_ = self.decoder_index_ + 1
        return decoder_command

    def get_filter_command(self, node):
        filter_command = ""
        for input_stream in node.get_input_stream_names():
            if "c_ffmpeg_decoder" in input_stream:
                input_filter = self.decoder_input_streams_[input_stream]
                filter_command = filter_command + input_filter
            else:
                filter_command = filter_command + "[{}]".format(input_stream)
        if "name" in node.option.keys():
            filter_command = filter_command + node.option["name"]
        if "para" in node.option.keys():
            filter_command = filter_command + "=" + self.escaping_param(
                node.option["para"])
        for output_stream in node.get_output_stream_names():
            filter_command = filter_command + "[{}]".format(output_stream)

        return filter_command

    def get_encoder_video_param(self, param):
        width = ""
        height = ""
        video_command = ""
        for key, value in param.items():
            if key == "codec":
                if value == 'h264':
                    codec = 'libx264'
                    pix_fmt = 'yuv420p'
                elif value == 'v265':
                    codec = 'libv265'
                    pix_fmt = 'yuv420p'
                elif value == 'jpg':
                    codec = 'mjpeg'
                    pix_fmt = 'yuvj444p'
                elif value == 'png':
                    codec = 'png'
                    pix_fmt = 'rgba'
                else:
                    codec = value
                    pix_fmt = ''
                if pix_fmt == '':
                    video_command = video_command + "-vcodec " + codec + " "
                else:
                    video_command = video_command + "-vcodec " + codec + " -pix_fmt " + pix_fmt + " "
            elif key == "width":
                width = param["width"]
            elif key == "height":
                height = param["height"]
            elif key == "bite_rate":
                video_command = video_command + "-" + "b:v" + " " + str(
                    value) + " "
            elif key == "max_fr":
                vsync = 'vfr'
                for k, v in param.items():
                    if k == "vsync":
                        vsync = v
                video_command = video_command + "-vsync" + " " + vsync + " -r" + " " + str(
                    value) + " "
            else:
                video_command = video_command + "-" + key + " " + str(
                    value) + " "
        if not (width == "" and height == ""):
            video_command = video_command + "-s " + str(width) + "x" + str(
                height) + " "
        return video_command

    def get_encoder_audio_param(self, param):
        audio_command = ""
        for key, value in param.items():
            if key == "codec":
                audio_command = audio_command + "-" + "acodec" + " " + str(
                    value) + " "
            elif key == "bit_rate":
                audio_command = audio_command + "-" + "b:a" + " " + str(
                    value) + " "
            elif key == "sample_rate":
                audio_command = audio_command + "-" + "ar" + " " + str(
                    value) + " "
            elif key == "channels":
                audio_command = audio_command + "-" + "ac" + " " + str(
                    value) + " "
            else:
                audio_command = audio_command + "-" + key + " " + str(
                    value) + " "
        return audio_command

    def get_encoder_mux_param(self, param):
        mux_command = ""
        for key, value in param.items():
            mux_command = mux_command + "-" + key + " " + str(value) + " "
        return mux_command

    def get_encoder_map(self, input_streams, index):
        map_string = ""
        if len(input_streams) <= index:
            return map_string
        if "encoder" in input_streams[index]:
            return map_string
        if "c_ffmpeg_decoder" in input_streams[index]:
            map_string = "-map " + self.decoder_input_streams_[
                input_streams[index]][1:-2] + " "
            return map_string
        else:
            map_string = "-map '[" + input_streams[index] + "]' "

            return map_string

    def get_encoder_command(self, node):
        # get video map
        video_map_string = self.get_encoder_map(node.get_input_stream_names(),
                                                0)
        video_command = ""
        if "video_params" in node.option.keys():
            video_command = self.get_encoder_video_param(
                node.option["video_params"])
        if video_command == "" and not video_map_string == "":
            video_command = "-vsync vfr  -r 120 "

        # get audio map
        audio_map_string = self.get_encoder_map(node.get_input_stream_names(),
                                                1)
        audio_command = ""
        if "audio_params" in node.option.keys():
            audio_command = self.get_encoder_audio_param(
                node.option["audio_params"])

        mux_command = ""
        if "mux_params" in node.option.keys():
            mux_command = self.get_encoder_mux_param(node.option["mux_params"])

        format_command = ""
        if "format" in node.option.keys():
            format_command = "-f " + node.option["format"] + " "
        else:
            format_command = "-f " + "mp4" + " "

        encoder_command = video_map_string + video_command + audio_map_string + audio_command + mux_command + \
                          format_command + node.option["output_path"] + " "
        return encoder_command

    def get_ffmpeg_command(self, graph_config):
        ffmpeg_path = os.getenv('FFMPEG_BIN_PATH')
        if ffmpeg_path is None:
            command = 'ffmpeg '
        else:
            command = ffmpeg_path + "/ffmpeg "
        # command = "ffmpeg "
        filter_command_list = []
        decoder_command_list = []
        encoder_command_list = []
        self.decoder_index_ = 0
        self.decoder_input_streams_ = {}
        for node in graph_config.get_nodes():
            md_name = node.get_module_info().get_name()
            if md_name == "c_ffmpeg_decoder":
                decoer_command = self.get_decoder_command(node)
                decoder_command_list.append(decoer_command)
            elif md_name == "c_ffmpeg_filter":
                filter_command = self.get_filter_command(node)
                filter_command_list.append(filter_command)
            elif md_name == "c_ffmpeg_encoder":
                encoder_command = self.get_encoder_command(node)
                encoder_command_list.append(encoder_command)

        for decoder_command in decoder_command_list:
            command = command + decoder_command
        if len(filter_command_list) > 0:
            command = command + ' -filter_complex "' + filter_command_list[0]
            for index in range(1, len(filter_command_list)):
                command = command + ";" + filter_command_list[index]
            command = command + '" '
        for encoder_command in encoder_command_list:
            command = command + encoder_command
        return command

    def is_valid_for_ffmpeg(self, graph_config):
        for node in graph_config.get_nodes():
            md_name = node.get_module_info().get_name()
            if not (md_name == "c_ffmpeg_decoder" or md_name
                    == "c_ffmpeg_filter" or md_name == "c_ffmpeg_encoder"):
                return False
        return True

    def run_command(self, command):
        os.system(command)
