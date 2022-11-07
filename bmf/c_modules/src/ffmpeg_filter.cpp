/*
 * Copyright 2023 Babit Authors
 *
 * This file is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * This file is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * Lesser General Public License for more details.
 */
#include "ffmpeg_filter.h"
#include <iostream>
#include <bmf/sdk/ffmpeg_helper.h>
#include <bmf/sdk/timestamp.h>
#include <bmf/sdk/log_buffer.h>
USE_BMF_SDK_NS


CFFFilter::CFFFilter(int node_id, JsonParam option) {
    node_id_ = node_id;
    num_input_streams_ = 0;
    num_output_streams_ = 0;
    b_graph_inited_ = false;
    filter_graph_ = NULL;
    is_inf_ = false;
    copy_ts_ = false;
    all_input_eof_ = false;
    all_output_eof_ = false;
    stream_start_time_ = AV_NOPTS_VALUE;
    stream_first_dts_ = AV_NOPTS_VALUE;

    option_ = option;
    avfilter_register_all();
}

CFFFilter::~CFFFilter() {
    clean();
}

bool CFFFilter::is_hungry(int input_stream_id) {
    if (input_cache_.count(input_stream_id) == 0 || input_cache_[input_stream_id].size() < 5 || filter_graph_ == NULL) {
        return true;
    }
    return false;
}

bool CFFFilter::is_infinity() {
    return is_inf_;
}

int CFFFilter::clean() {
    reset_check_mutex_.lock();
    if (filter_graph_) {
        delete filter_graph_;
        filter_graph_ = NULL;
    }
    num_input_streams_ = 0;
    num_output_streams_ = 0;
    g_desc_ = "";
    inpads_.clear();
    outpads_.clear();
    reset_check_mutex_.unlock();
    return 0;
}

int CFFFilter::close() {
    clean();
    return 0;
}

int CFFFilter::parse_filter(std::vector<JsonParam> &f_param, int idx, std::string &result) {
    std::vector<JsonParam> in_param;
    int ret;

    f_param[idx].get_object_list("inputs", in_param);

    //"stream" means the id of input/output stream
    //"pin" mainly used for the sequence number of the pin of the filter which the input stream connected to, and for output of the filter, the pin should always be zero now.

    for (int i = 0; i < in_param.size(); i++) {// [i0_0] [i1_0] [i2_0] [i3_0] ... [inputStream_Pin]
        if (in_param[i].has_key("stream") && in_param[i].has_key("pin")) {
            int stream_id;
            in_param[i].get_int("stream", stream_id);
            std::string tmp = "[i" + std::to_string(stream_id) + "_";
            int pin_id;
            in_param[i].get_int("pin", pin_id);
            tmp += std::to_string(pin_id) + "]";
            inpads_[idx][pin_id] = tmp;
        }
    }
    std::vector<JsonParam> out_param;
    f_param[idx].get_object_list("outputs", out_param);
    for (int i = 0; i < out_param.size(); i++) {// [o0_0] [o1_0] .... [outputStream_Pin]
        if (out_param[i].has_key("stream") && out_param[i].has_key("pin")) {
            int stream_id;
            out_param[i].get_int("stream", stream_id);
            std::string tmp = "[o" + std::to_string(stream_id) + "_";
            int pin_id;
            out_param[i].get_int("pin", pin_id);
            tmp += std::to_string(pin_id) + "]";
            outpads_[idx].push_back(tmp);
        }
    }

    std::map<int, std::string> inpad = inpads_[idx];
    for (auto pad = inpad.begin(); pad != inpad.end(); ++pad) {
        result += pad->second;
    }

    std::string res;
    ret = f_param[idx].get_string("name", res);
    result += res;

    if (res == "loop")
        is_inf_ = true;

    if (f_param[idx].has_key("para")) {
        ret = f_param[idx].get_string("para", res);
        if (res.find(',') != res.npos)
            res = "\'" + res + "\'";

        result += "=" + res;
    }

    std::vector<JsonParam> link_param;
    f_param[idx].get_object_list("links", link_param);
    for (int i = 0; i < link_param.size(); i++) {
        int fid, outpid;
        link_param[i].get_int("output_filter", fid);
        link_param[i].get_int("output_pin", outpid);
        std::string pad = "[";
        pad += std::to_string(idx) + "_to_" + std::to_string(fid) + "]";

        inpads_[fid][outpid] = pad;
        result += pad;
    }

    for (const auto &pad : outpads_[idx]) {
        result += pad;
    }

    return 0;
}

int CFFFilter::graph_descr(JsonParam &option, std::string &result) {
    int ret;

    if (!option.has_key("filters")) {
        //Log: No filter config
        BMFLOG_NODE(BMF_ERROR, node_id_) << "No filter config";
        return -1;
    }
    std::vector<JsonParam> filters_param;
    option.get_object_list("filters", filters_param);

    for (int i = 0; i < filters_param.size(); i++) {
        std::string res;

        if (i != 0)
            result += ";";
        parse_filter(filters_param, i, res);
        result += res;
    }
    return 0;
}

int CFFFilter::init_filtergraph() {
    int ret = 0;
    AVRational time_base;
    AVRational frame_rate;

    ret = graph_descr(option_, g_desc_);
    if (ret < 0)
        return ret;
    for (auto it = input_cache_.begin(); it != input_cache_.end(); it++) {
        AVFrame *frm = it->second.front();
        if (!frm)
            continue;
        config_[it->first].width = frm->width;
        config_[it->first].height = frm->height;
        config_[it->first].format = frm->format;
        config_[it->first].sample_aspect_ratio = frm->sample_aspect_ratio;
        config_[it->first].sample_rate = frm->sample_rate;
        config_[it->first].channels = frm->channels;
        config_[it->first].channel_layout = frm->channel_layout;

        config_[it->first].tb = (frm->width && frm->height) ? (AVRational){1, 25} : (AVRational){1, frm->sample_rate};

        if (frm->metadata) {
            AVDictionaryEntry *tag = NULL;
            while ((tag = av_dict_get(frm->metadata, "", tag, AV_DICT_IGNORE_SUFFIX))) {
                if (!strcmp(tag->key, "time_base")) {
                    std::string svalue = tag->value;
                    int pos = svalue.find(",");//"num,den"
                    if (pos > 0) {
                        AVRational r;
                        r.num = stoi(svalue.substr(0, pos));
                        r.den = stoi(svalue.substr(pos + 1));
                        config_[it->first].tb = r;
                    }
                }
                if (!strcmp(tag->key, "frame_rate")) {
                    std::string svalue = tag->value;
                    int pos = svalue.find(",");
                    if (pos > 0) {
                        AVRational frame_rate;
                        frame_rate.num = stoi(svalue.substr(0, pos));
                        frame_rate.den = stoi(svalue.substr(pos + 1));
                        config_[it->first].frame_rate = frame_rate;
                    }
                }
                if (!strcmp(tag->key, "start_time")) {
                    std::string svalue = tag->value;
                    stream_start_time_ = stol(svalue);
                }
                if (!strcmp(tag->key, "first_dts")) {
                    std::string svalue = tag->value;
                    stream_first_dts_ = stol(svalue);
                }
                if (!strcmp(tag->key, "stream_node_id")) {
                    std::string svalue = tag->value;
                    input_stream_node_[it->first] = stoi(svalue);
                }
                if (!strcmp(tag->key, "copyts"))
                    copy_ts_ = true;
            }
        }
    }

    filter_graph_ = new FilterGraph();
    std::map<int, FilterConfig> out_cfgs;
    ret = filter_graph_->config_graph(g_desc_, config_, out_cfgs);
    if (ret != 0)
        return ret;

    b_graph_inited_ = true;
    return 0;
}

Packet CFFFilter::convert_avframe_to_packet(AVFrame *frame, int index) {
    AVRational tb = av_buffersink_get_time_base(filter_graph_->buffer_sink_ctx_[index]);
    std::string s_tb = std::to_string(tb.num) + "," + std::to_string(tb.den);
    av_dict_set(&frame->metadata, "time_base", s_tb.c_str(), 0);

    if (frame->width > 0) {
        AVRational frame_rate = av_buffersink_get_frame_rate(filter_graph_->buffer_sink_ctx_[index]);
        if (frame_rate.num > 0 && frame_rate.den > 0)
            s_tb = std::to_string(frame_rate.num) + "," + std::to_string(frame_rate.den);
        else
            s_tb = "0,1";
        av_dict_set(&frame->metadata, "frame_rate", s_tb.c_str(), 0);

        AVRational sar = av_buffersink_get_sample_aspect_ratio(filter_graph_->buffer_sink_ctx_[index]);
        if (sar.num > 0 && sar.den > 0)
            s_tb = std::to_string(sar.num) + "," + std::to_string(sar.den);
        else
            s_tb = "0,1";
        av_dict_set(&frame->metadata, "sample_aspect_ratio", s_tb.c_str(), 0);

        std::string st;
        if (stream_start_time_ != AV_NOPTS_VALUE && num_input_streams_ == 1) {
            st = std::to_string(stream_start_time_);
            av_dict_set(&frame->metadata, "start_time", st.c_str(), 0);
        }
        if (stream_first_dts_ != AV_NOPTS_VALUE && num_input_streams_ == 1) {
            st = std::to_string(stream_first_dts_);
            av_dict_set(&frame->metadata, "first_dts", st.c_str(), 0);
        }
        if (orig_pts_time_cache_.size() > 0) {
            if (orig_pts_time_cache_.count(frame->coded_picture_number) > 0) {
                av_dict_set(&frame->metadata, "orig_pts_time", orig_pts_time_cache_[frame->coded_picture_number].c_str(), 0);
                orig_pts_time_cache_.erase(frame->coded_picture_number);
            }
        }
    }
    if (copy_ts_)
        av_dict_set(&frame->metadata, "copyts", "1", 0);

    if (frame->width > 0) {
	    auto video_frame = ffmpeg::to_video_frame(frame);
        video_frame.set_time_base(Rational(tb.num, tb.den));
        video_frame.set_pts(frame->pts);
        auto packet = Packet(video_frame);
        packet.set_timestamp(frame->pts * av_q2d(tb) * 1000000);
        return packet;
    } else {
        auto audio_frame = ffmpeg::to_audio_frame(frame);
        audio_frame.set_time_base(Rational(tb.num, tb.den));
        audio_frame.set_pts(frame->pts);
        auto packet = Packet(audio_frame);
        packet.set_timestamp(frame->pts * av_q2d(tb) * 1000000);
        return packet;
    }
}

std::string get_meta_info(AVFrame *temp_frame, std::string key) {
    if (temp_frame != NULL) {
        if (temp_frame->metadata) {
            AVDictionaryEntry *tag = NULL;
            while ((tag = av_dict_get(temp_frame->metadata, "", tag, AV_DICT_IGNORE_SUFFIX))) {
                if (!strcmp(tag->key, key.c_str())) {
                    std::string svalue = tag->value;
                    //int64_t stream_frame_number = stol(svalue);
                    //return stream_frame_number;
                    return svalue;
                }
            }
        }
    }
    return "";
}

int CFFFilter::get_cache_frame(int index, AVFrame *&frame, int &choose_index) {
    //from input cache, choose the first frame decode from the same node.
    int choose_node_id = -1;
    int64_t choose_node_frame_number = INT64_MAX;
    choose_index = index;
    if (input_stream_node_.count(index) > 0) {
        choose_node_id = input_stream_node_[index];
    } else {
        if (input_cache_[index].size() > 0) {
            frame = input_cache_[index].front();
            input_cache_[index].pop();
            choose_index = index;
            return 0;
        }
    }

    for (auto input_stream_node : input_stream_node_) {
        if (input_stream_node.second == choose_node_id) {
            int same_node_index = input_stream_node.first;
            if (input_cache_[same_node_index].size() > 0) {
                AVFrame *temp_frame = input_cache_[same_node_index].front();
                std::string svalue = get_meta_info(temp_frame, "stream_frame_number");
                int64_t stream_frame_number = svalue != "" ? stol(svalue) : -1;
                if (stream_frame_number != -1 && stream_frame_number < choose_node_frame_number) {
                    choose_node_frame_number = stream_frame_number;
                    choose_index = same_node_index;
                }

                svalue = get_meta_info(temp_frame, "orig_pts_time");
                if (svalue != "")
                    orig_pts_time_cache_[temp_frame->coded_picture_number] = svalue;
            }
        }
    }
    if (input_cache_[choose_index].size() > 0) {
        frame = input_cache_[choose_index].front();
        input_cache_[choose_index].pop();
        return 0;
    }

    return -1;
}

int CFFFilter::process_filter_graph(Task &task) {
    int ret = 0;
    if (input_cache_.size() < num_input_streams_) {
        return 0;
    }
    static int push_frame_number = 0;
    static std::map<int, int> push_frame_number_map;
    if (!b_graph_inited_) {
        if ((ret = init_filtergraph()) != 0)
            return ret;
        std::map<int, bool> init_push_frame;
        for (auto inputs : input_cache_) {
            int index = inputs.first;
            AVFrame *frame;
            int choose_index;

            while (init_push_frame.count(index) == 0) {
                if (get_cache_frame(index, frame, choose_index) >= 0) {
                    init_push_frame[choose_index] = true;
                    if (frame) {
                        push_frame_number++;
                        if (push_frame_number_map.count(choose_index) == 0) {
                            push_frame_number_map[choose_index] = 0;
                        }
                        push_frame_number_map[choose_index]++;
                        av_dict_free(&frame->metadata);
                    }
                    filter_graph_->push_frame(frame, choose_index);
                    if (frame) {
                        av_frame_free(&frame);
                    }
                }
            }
        }
    }
    int push_frame_flag = 0;
    if (num_input_streams_ > 0) {
        while (true) {
            std::map<int, std::vector<AVFrame *>> output_frames;
            int ret = filter_graph_->reap_filters(output_frames, push_frame_flag);
            push_frame_flag = 0;
            for (auto output_frame : output_frames) {
                for (int index = 0; index < output_frame.second.size(); index++) {
                    if (output_frame.second[index] == NULL) {
                        out_eof_[output_frame.first] = true;
                        continue;
                    }
                    static int conver_av_frame = 0;
                    conver_av_frame++;
                    auto packet = convert_avframe_to_packet(output_frame.second[index], output_frame.first);
                    av_frame_free(&(output_frame.second[index]));
                    task.fill_output_packet(output_frame.first, packet);
                }
            }
            if (check_finished()) {
                return 0;
            }
            if (ret >= 0) {
                continue;
            }
            if (ret != AVERROR(EAGAIN) && ret < 0) {
                return ret;
            }
            int index = filter_graph_->get_the_most_failed_nb_request();
            if (index >= 0) {
                AVFrame *frame;
                int choose_index;
                if (get_cache_frame(index, frame, choose_index) >= 0) {
                    if (frame) {
                        push_frame_number++;
                        if (push_frame_number_map.count(choose_index) == 0) {
                            push_frame_number_map[choose_index] = 0;
                        }
                        push_frame_number_map[choose_index]++;
                        av_dict_free(&frame->metadata);
                    }
                    filter_graph_->push_frame(frame, choose_index);
                    push_frame_flag = 1;
                    if (frame) {
                        av_frame_free(&frame);
                    }
                } else if (in_eof_[index]) {
                    filter_graph_->push_frame(NULL, index);
                    push_frame_flag = 1;
                } else {
                    break;
                }
            }
        }
    } else {
        // this is a source filter, so we get frame from sink for one time in a process.
        std::map<int, std::vector<AVFrame *>> output_frames;
        int ret = filter_graph_->reap_filters(output_frames, 0);
        for (auto output_frame : output_frames) {
            for (int index = 0; index < output_frame.second.size(); index++) {
                if (output_frame.second[index] == NULL) {
                    out_eof_[output_frame.first] = true;
                    continue;
                }
                auto packet = convert_avframe_to_packet(output_frame.second[index], output_frame.first);
                av_frame_free(&(output_frame.second[index]));
                task.fill_output_packet(output_frame.first, packet);
            }
        }
        return ret;
    }
    return 0;
}

bool CFFFilter::check_finished() {
    for (int i = 0; i < num_output_streams_; i++) {
        if (!out_eof_[i]) {
            return false;
        }
    }
    return true;
}

int CFFFilter::process(Task &task) {

    Packet packet;
    AVFrame *frame;
    bool b_eof = true;
    int ret = 0;

    //in case of possiblity to be used in dynamical remove/add, just return now
    if (num_input_streams_ != 0 && num_input_streams_ != task.get_inputs().size()) {
        BMFLOG_NODE(BMF_INFO, node_id_) << "the inputs number has changed, just return";
        return PROCESS_OK;
    }

    if (!num_input_streams_) {
        num_input_streams_ = task.get_inputs().size();
        if (in_eof_.size() == 0) {
            for (int i = 0; i < num_input_streams_; i++) {
                in_eof_.push_back(false);
            }
        }
    }
    if (!num_output_streams_) {
        num_output_streams_ = task.get_outputs().size();
        if (out_eof_.size() == 0) {
            for (int i = 0; i < num_output_streams_; i++) {
                out_eof_.push_back(false);
            }
        }
    }

    //cache task data to input_cache
    for (int index = 0; index < num_input_streams_; index++) {
        while (task.pop_packet_from_input_queue(index, packet)) {
            if (packet.timestamp() == BMF_EOF) {
                input_cache_[index].push(NULL);
                in_eof_[index] = true;
                continue;
            }

            if (packet.timestamp() == BMF_PAUSE) {
                BMFLOG_NODE(BMF_INFO, node_id_) << "Got BMF PAUSE in the filter";
                for (auto &outputs : task.get_outputs())
                    outputs.second->push(packet);
                continue;
            }
            if (packet.timestamp() == DYN_EOS) {
                BMFLOG_NODE(BMF_INFO, node_id_) << "Got DYN EOS in the filter";
                for (auto &outputs : task.get_outputs())
                    outputs.second->push(packet);
                return PROCESS_OK;
            }

            if (packet.is<VideoFrame>()) {
                auto& video_frame = packet.get<VideoFrame>();
                frame = ffmpeg::from_video_frame(video_frame, true);
            }
            else if (packet.is<AudioFrame>()) {
                auto& audio_frame = packet.get<AudioFrame>();
                frame = ffmpeg::from_audio_frame(audio_frame, true);
            }
            else {
                return PROCESS_ERROR;
            }
            input_cache_[index].push(frame);
            static int input_cache_number = 0;
            input_cache_number++;
        }
    }

    if (ret = process_filter_graph(task) < 0) {
        return ret;
    };

    if (check_finished()) {
        for (int index = 0; index < num_output_streams_; index++) {
            Packet packet = Packet::generate_eof_packet();
            task.fill_output_packet(index, packet);
        }
        task.set_timestamp(DONE);
        //clear input_cache
        for (auto input : input_cache_) {
            while (input.second.size() > 0) {
                AVFrame *frame = input.second.front();
                if (frame) {
                    av_frame_free(&frame);
                }
                input.second.pop();
            }
        }
    }

    return PROCESS_OK;
}

int CFFFilter::reset() {
    for (int i = 0; i < num_input_streams_; i++) {
        in_eof_[i] = false;
    }
    for (int i = 0; i < num_output_streams_; i++) {
        out_eof_[i] = false;
    }
    all_input_eof_ = false;
    all_output_eof_ = false;
    b_graph_inited_ = false;
    clean();
    return 0;
}
