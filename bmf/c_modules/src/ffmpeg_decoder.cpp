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

#include <unistd.h>
#include "ffmpeg_decoder.h"
#include <string>
#include <cstddef>
#include <bmf/sdk/video_frame.h>
#include <bmf/sdk/audio_frame.h>
#include <bmf/sdk/ffmpeg_helper.h>
#include <bmf/sdk/bmf_av_packet.h>
#include <bmf/sdk/log_buffer.h>
#include "libavutil/display.h"
#include "libavutil/timestamp.h"
#include <bmf/sdk/exception_factory.h>
#include <bmf/sdk/error_define.h>
using json = bmf_nlohmann::json;

#define BMF_CONV_FP(x) ((double) (x)) / (1 << 16)

USE_BMF_SDK_NS

CFFDecoder::CFFDecoder(int node_id, JsonParam option) {
    node_id_ = node_id;
    input_path_ = "";
    input_fmt_ctx_ = NULL;
    skip_frame_ = AVDISCARD_NONE;
    video_stream_ = NULL;
    audio_stream_ = NULL;
    decoded_frm_ = NULL;
    video_stream_index_ = -1;
    audio_stream_index_ = -1;
    video_decode_ctx_ = NULL;
    audio_decode_ctx_ = NULL;
    filter_graph_[0] = NULL;
    filter_graph_[1] = NULL;
    video_frame_count_ = 0;
    audio_frame_count_ = 0;
    refcount_ = 1;
    
    encrypted_ = false;
    audio_end_ = false;
    video_end_ = false;
    start_time_ = AV_NOPTS_VALUE;
    end_time_ = 0;
    last_pts_ = AV_NOPTS_VALUE;
    curr_pts_ = 0;
    last_ts_ = AV_NOPTS_VALUE;
    //duration_ = 0;
    idx_dur_ = -1;
    dur_end_[0] = false;
    dur_end_[1] = false;
    end_audio_time_ = LLONG_MAX;
    end_video_time_ = LLONG_MAX;
    fps_ = 25;//default when there's no correct framerate

    ist_[0] = ist_[1] = {0};
    ist_[0].max_pts = ist_[1].max_pts = INT64_MIN;
    ist_[0].min_pts = ist_[1].min_pts = INT64_MAX;
    ist_[0].max_frames = ist_[1].max_frames = INT64_MAX;
    ist_[0].decoding_needed = ist_[1].decoding_needed = true;
    ist_[0].filter_in_rescale_delta_last = ist_[1].filter_in_rescale_delta_last = AV_NOPTS_VALUE;

    stream_copy_av_stream_flag_[0] = false;
    stream_copy_av_stream_flag_[1] = false;

    extract_frames_fps_ = 0;
    extract_frames_device_ = "";
    stream_frame_number_ = 0;

    if (!LogBuffer::avlog_cb_set())
        av_log_set_level(40);
    /** @addtogroup DecM
     * @{
     * @arg loglevel: to set the loglevel of ffmpeg library
     * it can be "quiet","panic","fatal","error","warning","info","verbose","debug","trace"
     * @} */
    if (input_option_.has_key("loglevel")) {
        std::string log_level = "";
        input_option_.get_string("loglevel", log_level);
        if (!LogBuffer::avlog_cb_set()) {
            av_log_set_level(LogBuffer::infer_level(log_level));
            BMFLOG_NODE(BMF_INFO, node_id_) << "decode setting log level to: " << log_level;
        }
    }

    /** @addtogroup DecM
     * @{
     * @arg map_v: video stream index for decoder, exp. 0, which mean choose No.0 stream as video stream to be decode.
     * @} */
    if (option.has_key("map_v"))
        option.get_int("map_v", video_stream_index_);

    /** @addtogroup DecM
     * @{
     * @arg map_a: audio stream index for decoder, exp. 1, which mean choose No.1 stream as audio stream to be decode.
     * @} */
    if (option.has_key("map_a"))
        option.get_int("map_a", audio_stream_index_);

    /** @addtogroup DecM
     * @{
     * @arg start_time: decode start time in seconds, exp. 1, which mean just decode the frame after 1 second, similar as -ss in ffmpeg command.
     * @} */
    if (option.has_key("start_time")) {
        double opt;
        option.get_double("start_time", opt);
        start_time_ = (int64_t)(opt * AV_TIME_BASE);
    }
    /** @addtogroup DecM
     * @{
     * @arg end_time: decode end time, exp. 1, which mean just decode the frame before 1 second, just as -to in ffmpeg command.
     * @} */
    if (option.has_key("end_time")) {
        double opt;
        option.get_double("end_time", opt);
        end_time_ = (int64_t)(opt * AV_TIME_BASE);
    }
    
    /** @addtogroup DecM
     * @{
     * @arg durations: decode multiple group of duration frames/samples, such as [1.1, 4, 6.5, 9, 12.3, 15].
     * @} */
    if (option.has_key("durations")) {
        option.get_double_list("durations", durations_);
        if (durations_.size() > 0 && durations_.size() % 2 == 0) {
            auto min = durations_[0];
            bool is_valid = true;
            for(auto it:durations_) {
                if (it < min) {
                    BMFLOG_NODE(BMF_ERROR, node_id_) << "The durations are incorrect";
                    is_valid = false;
                    break;
                }
                min = it;
            }
            if (is_valid) {
                idx_dur_ = 0;
                //to prepare the first start time which overwrite start_time param
                start_time_ = (int64_t)(durations_[0] * AV_TIME_BASE);
                end_time_ = 0;
            }
        } else
            BMFLOG_NODE(BMF_ERROR, node_id_) << "The durations timestamp number should be even";

        for(auto it:durations_) {
            BMFLOG_NODE(BMF_DEBUG, node_id_) << "durations " << it;
        }
    }

    /** @addtogroup DecM
     * @{
     * @arg fps: decode the frame as the fps set .
     * @} */
    if (option.has_key("fps"))
        option.get_int("fps", fps_);
    
    /** @addtogroup DecM
     * @{
     * @arg video_time_base: video stream time base, exp. 1/1000, set the video stream timebase as 1/1000.
     * @} */
    if (option.has_key("video_time_base"))
        option.get_string("video_time_base", video_time_base_string_);
    
    /** @addtogroup DecM
     * @{
     * @arg skip_frame: skip frame, exp. 32, make decoder discard processing depending on the option value, just as -skip_frame in ffmpeg commnad.
     * @} */
    if (option.has_key("skip_frame")) {
        int tmp;
        option.get_int("skip_frame", tmp);
        skip_frame_ = (enum AVDiscard) tmp;
    }
    /** @addtogroup DecM
     * @{
     * @arg video_codec: video codec name, exp. libx264, set the specific codec for video stream.
     * it will be stream copy when it set to be "copy"
     * @} */
    if (option.has_key("video_codec")) {
        std::string tmp;
        option.get_string("video_codec", tmp);
        if (tmp == "copy")
            ist_[0].decoding_needed = false;
        else
            video_codec_name_ = tmp;
    }
    
    /** @addtogroup DecM
     * @{
     * @arg overlap_time, which is used in decode live stream, if the live stream cut off, 
     * if the next packet pts is overlap smaller than overlap time, we will remove the overlap packet.
     * default value is 10 
     * @} */
    if (option.has_key("overlap_time")) {
        double tmp;
        option.get_double("overlap_time", tmp);
        overlap_time_ = tmp * AV_TIME_BASE;
    } else {
        overlap_time_ = AV_TIME_BASE * 10;
    }
    
    /** @addtogroup DecM
     * @{
     * @arg cut_off_time, which is used in decode live stream ,if the live stream cut off,
     * when the next packet pts is larger than last pts + cut_off_time, we will adjust pts to avoid large cut off.
     * else we use the packet pts.  
     * @} */
    if (option.has_key("cut_off_time")) {
        double tmp;
        option.get_double("cut_off_time", tmp);
        cut_off_time_ = tmp * AV_TIME_BASE;
    } else {
        cut_off_time_ = AV_TIME_BASE * 10;
    }

    /** @addtogroup DecM
     * @{
     * @arg cut_off_interval.which is used in decode live stream ,if the live stream cut off,
     * when the next packet pts is larger than last pts + cut_off_time, we will adjust pts to avoid large cut off.
     * else we use the packet pts.  
     * @} */
    if (option.has_key("cut_off_interval")) {
        double tmp;
        option.get_double("cut_off_interval", tmp);
        cut_off_interval_ = tmp * AV_TIME_BASE;
    } else {
        cut_off_interval_ = AV_TIME_BASE / 30;
    }

    /** @addtogroup DecM
     * @{
     * @arg vframes: set the number of video frames to output
     * @} */
    if (option.has_key("vframes"))
        option.get_long("vframes", ist_[0].max_frames);
    /** @addtogroup DecM
     * @{
     * @arg aframes: set the number of audio frames to output
     * @} */
    if (option.has_key("aframes"))
        option.get_long("aframes", ist_[1].max_frames);
    assert(ist_[0].max_frames >= 0 && ist_[1].max_frames >= 0);

    /** @addtogroup DecM
     * @{
     * @arg hwaccel: hardware accelete exp. cuda.
     * @arg extract_frames: support extract frames with given fps and device.
     * @} */
    if (option.has_key("video_params")) {
        JsonParam video_params;
        option.get_object("video_params", video_params);
        if (video_params.has_key("hwaccel")) {
            video_params.get_string("hwaccel", hwaccel_str_);
        }
        if (video_params.has_key("hwaccel_check")) {
            video_params.get_int("hwaccel_check", hwaccel_check_);
        }
        if (video_params.has_key("extract_frames")) {
            JsonParam extract_frames_params;
            video_params.get_object("extract_frames", extract_frames_params);
            if (extract_frames_params.has_key("fps")) {
                extract_frames_params.get_double("fps", extract_frames_fps_);
            }
            if (extract_frames_params.has_key("device")) {
                extract_frames_params.get_string("device",extract_frames_device_);
            }
        }
    }

    /** @addtogroup DecM
     * @{
     * @arg audio_codec: audio codec name, exp. aac, set the specific codec for audio stream.
     * @} */
    if (option.has_key("audio_codec")) {
        std::string tmp;
        option.get_string("audio_codec", tmp);
        if (tmp == "copy")
            ist_[1].decoding_needed = false;
        else
            audio_codec_name_ = tmp;
    }

    /** @addtogroup DecM
     * @{
     * @arg dec_params: set the decode codec parameters, such as "threads": 1
     * @} */
    AVDictionary *opts = dec_opts_;
    if (option.has_key("dec_params")) {
        option.get_object("dec_params", dec_params_);
        std::vector<std::pair<std::string, std::string>> params;
        dec_params_.get_iterated(params);
        for (int i = 0; i < params.size(); i++) {
            av_dict_set(&opts, params[i].first.c_str(), params[i].second.c_str(), 0);
        }
    }
    if (option.has_key("decryption_key")) {
        std::string decryption_key;
        option.get_string("decryption_key", decryption_key);
        av_dict_set(&opts, "decryption_key", decryption_key.c_str(), 0);
        av_dict_set(&opts, "scan_all_pmts", "1", 0);
        encrypted_ = true;
    }

    /** @addtogroup DecM
     * @{
     * @arg autorotate: to enable/disable autorotate for the input video if needed, it's enabled by default
     * @} */
    if (option.has_key("autorotate")) {
        int autorotate = 1;
        option.get_int("autorotate", autorotate);
        if (autorotate == 0) {
            auto_rotate_flag_ = false;
        }
    }

    decoded_frm_ = av_frame_alloc();
    if (!decoded_frm_) {
        BMFLOG_NODE(BMF_ERROR, node_id_) << "Could not allocate frame";
        return;
    }

    /** @addtogroup DecM
     * @{
     * @arg s: video size, exp. "1280:720".
     * @} */
    if (option.has_key("s")) {
        std::string tmp;
        option.get_string("s", tmp);
        av_dict_set(&opts, "video_size", tmp.c_str(), 0);
    }
    /** @addtogroup DecM
     * @{
     * @arg pix_fmt: pixel format, exp. "rgba".
     * @} */
    if (option.has_key("pix_fmt")) {
        std::string tmp;
        option.get_string("pix_fmt", tmp);
        av_dict_set(&opts, "pixel_format", tmp.c_str(), 0);
    }
    /** @addtogroup DecM
     * @{
     * @arg input_path: decode input file,exp. "1.mp4".
     * @} */
    if (option.has_key("input_path")) {
        if (!init_done_) {
            option.get_string("input_path", input_path_);
            init_input(opts);
        }
    } else {
        has_input_ = true;
    }
    /** @addtogroup DecM
     * @{
     * @arg push_raw_stream: enable raw stream push mode, exp. 1
     * @} */
    if (option.has_key("push_raw_stream")) {
        option.get_int("push_raw_stream", push_raw_stream_);
        if (push_raw_stream_) {
            if (!option.has_key("video_codec") && !option.has_key("audio_codec"))
                throw std::runtime_error("Push raw stream requires either 'video_codec' or 'video_codec' in option");
            if (option.has_key("video_codec") && !option.has_key("video_time_base"))
                throw std::runtime_error("Missing 'video_time_base' in decoder option");
            if (option.has_key("audio_codec") && (!option.has_key("channels") || !option.has_key("sample_rate")))
                throw std::runtime_error("Missing 'channels' or 'sample_rate' in decoder option");
        }
    } else {
        push_raw_stream_ = 0;
    }

    /** @addtogroup DecM
     * @{
     * @arg channels: audio channels (required for audio push mode)
     * @} */
    if (option.has_key("channels"))
        option.get_int("channels", push_audio_channels_);

    /** @addtogroup DecM
     * @{
     * @arg sample_rate: audio sample rate (required for audio push mode)
     * @} */
    if (option.has_key("sample_rate"))
        option.get_int("sample_rate", push_audio_sample_rate_);

    /** @addtogroup DecM
     * @{
     * @arg sample_fmt: audio sample format (used for audio push mode - optional)
     * @} */
    if (option.has_key("sample_fmt"))
        option.get_int("sample_fmt", push_audio_sample_fmt_);

    return;
}

CFFDecoder::~CFFDecoder() {
    std::lock_guard<std::mutex> lock(mutex_);
    clean();
    BMFLOG_NODE(BMF_INFO, node_id_) << "video frame decoded:" << ist_[0].frame_decoded;
    BMFLOG_NODE(BMF_INFO, node_id_) << "audio frame decoded:" << ist_[1].frame_decoded
                                     << ", sample decoded:" << ist_[1].sample_decoded;
}


int copy_simple_frame(AVFrame* frame) {
    
    AVFrame tmp;
    int ret;

    if (!frame->buf[0])
        return AVERROR(EINVAL);

    memset(&tmp, 0, sizeof(tmp));
    tmp.format         = frame->format;
    tmp.width          = frame->width;
    tmp.height         = frame->height;
    tmp.channels       = frame->channels;
    tmp.channel_layout = frame->channel_layout;
    tmp.nb_samples     = frame->nb_samples;

    if (frame->hw_frames_ctx)
        ret = av_hwframe_get_buffer(frame->hw_frames_ctx, &tmp, 0);
    else
        ret = av_frame_get_buffer(&tmp, 0);
    if (ret < 0)
        return ret;
    //need to reset frame info for which the av_hwframe_get_buffer may change the width or height according to the hw_frames_ctx.
    tmp.format         = frame->format;
    tmp.width          = frame->width;
    tmp.height         = frame->height;
    tmp.channels       = frame->channels;
    tmp.channel_layout = frame->channel_layout;
    tmp.nb_samples     = frame->nb_samples;

    ret = av_frame_copy(&tmp, frame);
    if (ret < 0) {
        av_frame_unref(&tmp);
        return ret;
    }

    ret = av_frame_copy_props(&tmp, frame);
    if (ret < 0) {
        av_frame_unref(&tmp);
        return ret;
    }

    av_frame_unref(frame);

    *frame = tmp;
    if (tmp.data == tmp.extended_data)
        frame->extended_data = frame->data;
    return 0;
}


int CFFDecoder::codec_context(int *stream_idx,
                              AVCodecContext **dec_ctx, AVFormatContext *fmt_ctx, enum AVMediaType type) {
    int ret, stream_index;
    AVStream *st;
    AVCodec *dec = NULL;
    AVDictionary *opts = dec_opts_;
    int stream_id = -1;

    ret = av_find_best_stream(fmt_ctx, type, *stream_idx, -1, NULL, 0);
    if (ret < 0) {
        BMFLOG_NODE(BMF_ERROR, node_id_)
                    << "Could not find " << av_get_media_type_string(type) 
                    << " stream in input file '" << input_path_.c_str() << "'";
        return ret;
    } else {
        stream_index = ret;
        AVStream *st;
        st = fmt_ctx->streams[stream_index];
        if (type == AVMEDIA_TYPE_VIDEO) {
            if (video_codec_name_.empty()) {
                dec = avcodec_find_decoder(st->codecpar->codec_id);
            } else {
                dec = avcodec_find_decoder_by_name(video_codec_name_.c_str());
                st->codecpar->codec_id = dec->id;
            }
        } else if (type == AVMEDIA_TYPE_AUDIO) {
            if (audio_codec_name_.empty()) {
                dec = avcodec_find_decoder(st->codecpar->codec_id);
            } else {
                dec = avcodec_find_decoder_by_name(audio_codec_name_.c_str());
                st->codecpar->codec_id = dec->id;
            }
        }
        if (!dec) {
            BMFLOG_NODE(BMF_ERROR, node_id_)
                << "Failed to find " << av_get_media_type_string(type) << " codec";
            return AVERROR(EINVAL);
        }
        *dec_ctx = avcodec_alloc_context3(dec);
        if (!*dec_ctx) {
            BMFLOG_NODE(BMF_ERROR, node_id_) 
                << "Failed to allocate the " << av_get_media_type_string(type) << " codec context";
            return AVERROR(ENOMEM);
        }

        // Copy codec parameters from input stream to output codec context
        if ((ret = avcodec_parameters_to_context(*dec_ctx, st->codecpar)) < 0) {
            BMFLOG_NODE(BMF_ERROR, node_id_)
                            << "Failed to copy " << av_get_media_type_string(type) 
                            << " codec parameters to decoder context";
            return ret;
        }
        (*dec_ctx)->pkt_timebase = st->time_base;
        av_dict_set(&opts, "refcounted_frames", refcount_ ? "1" : "0", 0);
        if (!dec_params_.has_key("threads"))
            av_dict_set(&opts, "threads", "auto", 0);
        else {
            std::string td;
            dec_params_.get_string("threads", td);
            av_dict_set(&opts, "threads", td.c_str(), 0);
        }

        if (hwaccel_str_ == "cuda" && type == AVMEDIA_TYPE_VIDEO) {
            if (hwaccel_check_ == 0) {
                av_hwdevice_ctx_create(&((*dec_ctx)->hw_device_ctx), AV_HWDEVICE_TYPE_CUDA, NULL, NULL, 1);
            } else {
                if ((*dec_ctx)->has_b_frames < 2)
                    av_hwdevice_ctx_create(&((*dec_ctx)->hw_device_ctx), AV_HWDEVICE_TYPE_CUDA, NULL, NULL, 1);
            }
        }
        if ((ret = avcodec_open2(*dec_ctx, dec, &opts)) < 0) {
            BMFLOG_NODE(BMF_ERROR, node_id_) << "Failed to open " << av_get_media_type_string(type) << " codec";
            return ret;
        }
        *stream_idx = stream_index;
    }

    return 0;
}

int CFFDecoder::init_filtergraph(int index, AVFrame *frame) {
    std::string graph_descr = "";
    std::string head_descr = "[i0_0]";
    std::string tail_descr = "[o0_0]";
    double ts_offset = 0;
    double current_duration = 0;

    //if (copy_ts) {
    //    tsoffset = f->start_time == AV_NOPTS_VALUE ? 0 : f->start_time;
    //    if (!start_at_zero && f->ctx->start_time != AV_NOPTS_VALUE)
    //        tsoffset += f->ctx->start_time;
    //}
    if (durations_.size() > 0) {
        current_duration = (durations_[idx_dur_ + 1] - durations_[idx_dur_]);
        if (idx_dur_ > 0) {
            ts_offset = durations_[idx_dur_] + (double)(ts_offset_) / AV_TIME_BASE;
            printf("ts offset for trim: %f\n", ts_offset);
        }
    }

    if (start_time_ != AV_NOPTS_VALUE) {//accurate seek by default
        if (index == 0)
            graph_descr += "trim=starti=" + std::to_string(ts_offset);
        else if (index == 1)
            graph_descr += "atrim=starti=" + std::to_string(ts_offset);

        if (current_duration > 0) {
            graph_descr += ":durationi=" + std::to_string(current_duration);
        }
    }

    if (auto_rotate_flag_ && index == 0) {
        std::string rotate_desc = "";
        get_rotate_desc(rotate_desc);
        if (!rotate_desc.empty()) {
            if (graph_descr != "")
                graph_descr += "," + rotate_desc;
            else
                graph_descr = rotate_desc;
        }
    }
    if (graph_descr == "")
        return 0;
    else
        graph_descr = head_descr + graph_descr + tail_descr;

    filter_graph_[index] = new FilterGraph();
    FilterConfig fg_config;
    std::map<int, FilterConfig> in_cfgs;
    std::map<int, FilterConfig> out_cfgs;
    if (index == 0) {
        fg_config.width = frame->width;
        fg_config.height = frame->height;
        fg_config.format = frame->format;
        fg_config.sample_aspect_ratio = frame->sample_aspect_ratio;
        fg_config.tb = video_stream_->time_base;
        AVRational frame_rate = av_guess_frame_rate(input_fmt_ctx_, video_stream_, NULL);
        if (frame_rate.num > 0 && frame_rate.den > 0)
            fg_config.frame_rate = frame_rate;
    }
    if (index == 1) {
        fg_config.sample_rate = frame->sample_rate;
        fg_config.format = frame->format;
        fg_config.channels = frame->channels;
        fg_config.channel_layout = frame->channel_layout;
        fg_config.tb = audio_stream_->time_base;
    }
    in_cfgs[0] = fg_config;
    return filter_graph_[index]->config_graph(graph_descr, in_cfgs, out_cfgs);
}

int CFFDecoder::get_rotate_desc(std::string &filter_desc) {
    if (video_stream_ == NULL) {
        return 0;
    }
    uint8_t *displaymatrix = av_stream_get_side_data(video_stream_,
                                                     AV_PKT_DATA_DISPLAYMATRIX, NULL);
    double theta = 0;
    if (displaymatrix) {
        double rotation, scale[2];
        int32_t *matrix = (int32_t *) displaymatrix;
        scale[0] = hypot(BMF_CONV_FP(matrix[0]), BMF_CONV_FP(matrix[3]));
        scale[1] = hypot(BMF_CONV_FP(matrix[1]), BMF_CONV_FP(matrix[4]));
        if (scale[0] == 0.0 || scale[1] == 0.0) {
            theta = 0;
        } else {
            theta = atan2(BMF_CONV_FP(matrix[1]) / scale[1],
                          BMF_CONV_FP(matrix[0]) / scale[0]) * 180 / M_PI;
        }
    }

    theta -= 360 * floor(theta / 360 + 0.9 / 360);
    if (fabs(theta - 90) < 1.0) {
        filter_desc = "transpose=clock";
    } else if (fabs(theta - 180) < 1.0) {
        filter_desc = "hflip[0_0];[0_0]vflip";
    } else if (fabs(theta - 270) < 1.0) {
        filter_desc = "transpose=cclock";
    } else if (fabs(theta) > 1.0) {
        char rotate_buf[64];
        snprintf(rotate_buf, sizeof(rotate_buf), "%f*PI/180", theta);
        std::string temp(rotate_buf);
        filter_desc = "rotate=" + temp;
    }
    return 0;
}

int CFFDecoder::init_input(AVDictionary *options) {
    init_done_ = true;
    int ret;

    if (input_path_.empty()) {
        ret = avformat_open_input(&input_fmt_ctx_, NULL, NULL, &options);
        if (ret < 0) {
            std::string msg = "avformat_open_input failed: " + error_msg(ret);
            BMF_Error(BMF_TranscodeError, msg.c_str());
        }
    } else {
        ret = avformat_open_input(&input_fmt_ctx_, input_path_.c_str(), NULL, &options);
        if (ret < 0) {
            std::string msg = "avformat_open_input failed: " + error_msg(ret);
            BMF_Error(BMF_TranscodeError, msg.c_str());
        }
    }
    if ((ret = avformat_find_stream_info(input_fmt_ctx_, NULL)) < 0) {
        if (ret < 0) {
            std::string msg = "avformat_find_stream_info failed: " + error_msg(ret);
            BMF_Error(BMF_TranscodeError, msg.c_str());
        }
    }

    int64_t timestamp = (start_time_ == AV_NOPTS_VALUE) ? 0 : start_time_;
    if (input_fmt_ctx_->start_time != AV_NOPTS_VALUE)
        timestamp += input_fmt_ctx_->start_time;
    if (start_time_ != AV_NOPTS_VALUE) {
        int64_t seek_timestamp = timestamp;
        if (!(input_fmt_ctx_->iformat->flags & AVFMT_SEEK_TO_PTS)) {
            int dts_heuristic = 0;
            for (int i = 0; i < input_fmt_ctx_->nb_streams; i++) {
                const AVCodecParameters *par = input_fmt_ctx_->streams[i]->codecpar;
                if (par->video_delay) {
                    dts_heuristic = 1;
                    break;
                }
            }
            if (dts_heuristic) {
                seek_timestamp -= 3 * AV_TIME_BASE / 23;
            }
        }
        ret = avformat_seek_file(input_fmt_ctx_, -1, INT64_MIN, seek_timestamp,
                                 seek_timestamp, 0);
        if (ret < 0) {
            av_log(NULL, AV_LOG_WARNING, "%s: could not seek to position %0.3f\n",
                   input_path_.c_str(), (double)timestamp / AV_TIME_BASE);
        }
    }
    ts_offset_ = -timestamp;

    if (codec_context(&video_stream_index_, &video_decode_ctx_, input_fmt_ctx_, AVMEDIA_TYPE_VIDEO) >= 0) {
        video_stream_ = input_fmt_ctx_->streams[video_stream_index_];
        if (end_time_ > 0 && !encrypted_) {
            end_video_time_ = av_rescale_q((end_time_ + ts_offset_), AV_TIME_BASE_Q, video_stream_->time_base);
        }
        video_decode_ctx_->skip_frame = skip_frame_;
    }
    ist_[0].next_dts = AV_NOPTS_VALUE;
    ist_[0].next_pts = AV_NOPTS_VALUE;

    if (codec_context(&audio_stream_index_, &audio_decode_ctx_, input_fmt_ctx_, AVMEDIA_TYPE_AUDIO) >= 0) {
        audio_stream_ = input_fmt_ctx_->streams[audio_stream_index_];
        if (end_time_ > 0 && !encrypted_) {
            end_audio_time_ = av_rescale_q((end_time_ + ts_offset_), AV_TIME_BASE_Q, audio_stream_->time_base);
        }
    }
    ist_[1].next_dts = AV_NOPTS_VALUE;
    ist_[1].next_pts = AV_NOPTS_VALUE;

    if (!LogBuffer::avlog_cb_set())
        av_dump_format(input_fmt_ctx_, 0, input_path_.c_str(), 0);
    if (!audio_stream_ && !video_stream_) {
        BMF_Error(BMF_TranscodeError, "Could not find audio or video stream in the input");
    }

    return 0;
}

Packet CFFDecoder::generate_video_packet(AVFrame *frame)
{
    AVRational out_tb;
    if (filter_graph_[0])
        out_tb = av_buffersink_get_time_base(filter_graph_[0]->buffer_sink_ctx_[0]);
    else if (video_stream_)
        out_tb = video_stream_->time_base;

    if (!push_raw_stream_) {
        std::string s_tb = std::to_string(out_tb.num) + "," + std::to_string(out_tb.den);
        av_dict_set(&frame->metadata, "time_base", s_tb.c_str(), 0);
    } else
        av_dict_set(&frame->metadata, "time_base", video_time_base_string_.c_str(), 0);
    frame->pict_type = AV_PICTURE_TYPE_NONE;

    AVRational frame_rate;
    if (filter_graph_[0])
        frame_rate = av_buffersink_get_frame_rate(filter_graph_[0]->buffer_sink_ctx_[0]);
    else if (video_stream_)
        frame_rate = av_guess_frame_rate(input_fmt_ctx_, video_stream_, NULL);
    std::string s_fr;
    if (frame_rate.num && frame_rate.den)
        s_fr = std::to_string(frame_rate.num) + "," + std::to_string(frame_rate.den);
    else
        s_fr = "0,1";
    av_dict_set(&frame->metadata, "frame_rate", s_fr.c_str(), 0);

    if (video_stream_ && video_stream_->start_time != AV_NOPTS_VALUE) {
        std::string st = std::to_string(video_stream_->start_time);
        av_dict_set(&frame->metadata, "start_time", st.c_str(), 0);
    }
    if (video_stream_ && video_stream_->first_dts != AV_NOPTS_VALUE) {
        std::string f_dts = std::to_string(video_stream_->first_dts);
        av_dict_set(&frame->metadata, "first_dts", f_dts.c_str(), 0);
    }

    std::string stream_node_id_str = std::to_string(node_id_);
    av_dict_set(&frame->metadata, "stream_node_id", stream_node_id_str.c_str(), 0);

    stream_frame_number_++;
    std::string stream_frame_number_str = std::to_string(stream_frame_number_);
    av_dict_set(&frame->metadata, "stream_frame_number", stream_frame_number_str.c_str(), 0);

    auto video_frame = ffmpeg::to_video_frame(frame, true);
    video_frame.set_pts(frame->pts);
    if(!push_raw_stream_){
        video_frame.set_time_base(Rational(out_tb.num, out_tb.den));
    }
    else{
        video_frame.set_time_base(Rational(video_time_base_.num, video_time_base_.den));
    }
    
    auto packet = Packet(video_frame);
    if (!push_raw_stream_)
        packet.set_timestamp(frame->pts * av_q2d(video_stream_->time_base) * 1000000);
    else
        packet.set_timestamp(frame->pts * av_q2d(video_time_base_) * 1000000);

    av_frame_free(&(frame));
    return packet;
}

Packet CFFDecoder::generate_audio_packet(AVFrame *frame)
{
    AVRational out_tb;
    if (!push_raw_stream_ && filter_graph_[1])
        out_tb = av_buffersink_get_time_base(filter_graph_[1]->buffer_sink_ctx_[0]);
    else
        out_tb = (AVRational){1, audio_decode_ctx_->sample_rate};
    std::string s_tb = std::to_string(out_tb.num) + "," + std::to_string(out_tb.den);
    av_dict_set(&frame->metadata, "time_base", s_tb.c_str(), 0);

    std::string stream_node_id_str = std::to_string(node_id_);
    av_dict_set(&frame->metadata, "stream_node_id", stream_node_id_str.c_str(), 0);

    stream_frame_number_++;
    std::string stream_frame_number_str = std::to_string(stream_frame_number_);
    av_dict_set(&frame->metadata, "stream_frame_number", stream_frame_number_str.c_str(), 0);

    auto audio_frame = ffmpeg::to_audio_frame(frame, true);
    audio_frame.set_time_base(Rational(out_tb.num, out_tb.den));
    audio_frame.set_pts(frame->pts);

    auto packet = Packet(audio_frame);
    packet.set_timestamp(frame->pts * av_q2d(out_tb) * 1000000);

    av_frame_free(&(frame));
    return packet;
}

static int hwaccel_retrieve_data(AVFrame *input, enum AVPixelFormat output_format)
{
    AVFrame *output = NULL;
    int err;
    if (input->format == output_format) {
        // Nothing to do.
        return 0;
    }

    output = av_frame_alloc();
    if (!output)
        return AVERROR(ENOMEM);

    output->format = output_format;

    err = av_hwframe_transfer_data(output, input, 0);
    if (err < 0) {
        goto fail;
    }

    err = av_frame_copy_props(output, input);
    if (err < 0) {
        av_frame_unref(output);
        goto fail;
    }

    av_frame_unref(input);
    av_frame_move_ref(input, output);
    av_frame_free(&output);

    return 0;

fail:
    av_frame_free(&output);
    return err;
}

int CFFDecoder::extract_frames(AVFrame *frame,std::vector<AVFrame*> &output_frames) {
    AVRational frame_rate;
    if (filter_graph_[0])
        frame_rate = av_buffersink_get_frame_rate(filter_graph_[0]->buffer_sink_ctx_[0]);
    else if (video_stream_)
        frame_rate = av_guess_frame_rate(input_fmt_ctx_, video_stream_, NULL);

    // std::vector<AVFrame*> output_frames;
    if (extract_frames_fps_ != 0) {
        std::string extract_frames_fps_str = std::to_string(extract_frames_fps_);
        AVRational video_frame_rate;
        av_parse_video_rate(&video_frame_rate, extract_frames_fps_str.c_str());
        AVRational temp_time_base = {video_frame_rate.den, video_frame_rate.num};
        if (video_sync_ == NULL) {
            video_sync_ = std::make_shared<VideoSync>(video_stream_->time_base, temp_time_base, 
                                                                        frame_rate, video_frame_rate, video_stream_->start_time, video_stream_->first_dts, VSYNC_VFR, 0);
        }
        video_sync_->process_video_frame(frame, output_frames, ist_[0].frame_number);
        av_frame_free(&frame);
        for (int i =0;i<output_frames.size();i++) {
            output_frames[i]->pts = av_rescale_q(output_frames[i]->pts,temp_time_base,video_stream_->time_base);
        }
    } else
        output_frames.push_back(frame);

    for (int i =0; i<output_frames.size(); i++) {
        if (extract_frames_device_ == "CPU") {
            if (output_frames[i]->hw_frames_ctx) {
                enum AVPixelFormat output_pix_fmt;
                AVHWFramesContext* hw_frame_ctx = (AVHWFramesContext*)(output_frames[i]->hw_frames_ctx->data);
                output_pix_fmt = hw_frame_ctx->sw_format;
                hwaccel_retrieve_data(output_frames[i], output_pix_fmt);
            }
        }
        if (extract_frames_device_ == "CUDA") {
            enum AVPixelFormat output_pix_fmt = AV_PIX_FMT_CUDA;
            hwaccel_retrieve_data(output_frames[i], output_pix_fmt);
        }
    }
    return 0;
}

int CFFDecoder::handle_output_data(Task &task, int index, AVPacket *pkt, bool eof, bool repeat, int got_output) {
    AVFrame *frame;
    int64_t best_effort_timestamp;
    int64_t duration_dts = 0;
    int64_t duration_pts = 0;
    AVStream *stream = (index == 0) ? video_stream_ : audio_stream_;
    InputStream *ist = &ist_[index];

    if (ist->frame_number >= ist->max_frames) {
        index == 0 ? video_end_ = true : audio_end_ = true;
        Packet packet = Packet::generate_eof_packet();
        assert(packet.timestamp_ == BMF_EOF);
        if (task.get_outputs().find(index) != task.get_outputs().end() && file_list_.size() == 0)
            task.get_outputs()[index]->push(packet);

        push_data_flag_ = true;
        return 0;
    }

    if (!eof) {
        if (index == 0) {
            if (got_output) {
                best_effort_timestamp = decoded_frm_->best_effort_timestamp;
                duration_pts = decoded_frm_->pkt_duration;

                //if (ist->framerate.num) // force the fps
                //    best_effort_timestamp = ist->cfr_next_pts++;

                if (pkt && pkt->size == 0 && best_effort_timestamp == AV_NOPTS_VALUE && ist->nb_dts_buffer > 0) {
                    best_effort_timestamp = ist->dts_buffer[0];

                    BMFLOG_NODE(BMF_INFO, node_id_) << "nb_dts_buffer was used when eof of decode for best_effort_timestamp:"
                              << best_effort_timestamp;
                    for (int i = 0; i < ist->nb_dts_buffer - 1; i++)
                        ist->dts_buffer[i] = ist->dts_buffer[i + 1];
                    ist->nb_dts_buffer--;
                }

                if (stream && best_effort_timestamp != AV_NOPTS_VALUE) {
                    int64_t ts = av_rescale_q(decoded_frm_->pts = best_effort_timestamp,
                                              stream->time_base, AV_TIME_BASE_Q);

                    if (ts != AV_NOPTS_VALUE)
                        ist->next_pts = ist->pts = ts;
                }
            }

            if (!repeat || !pkt || (pkt && pkt->size == 0) || got_output) {
                if (stream && pkt && pkt->size != 0 && pkt->duration) {
                    duration_dts = av_rescale_q(pkt->duration, stream->time_base, AV_TIME_BASE_Q);
                } else if (stream && video_decode_ctx_->framerate.num != 0 && video_decode_ctx_->framerate.den != 0) {
                    int ticks= av_stream_get_parser(stream) ? av_stream_get_parser(stream)->repeat_pict+1 : video_decode_ctx_->ticks_per_frame;
                    duration_dts = ((int64_t)AV_TIME_BASE *
                                    video_decode_ctx_->framerate.den * ticks) /
                                    video_decode_ctx_->framerate.num / video_decode_ctx_->ticks_per_frame;
                }

                if (ist->dts != AV_NOPTS_VALUE && duration_dts) {
                    ist->next_dts += duration_dts;
                } else
                    ist->next_dts = AV_NOPTS_VALUE;
            }

            if (got_output) {
                if (duration_pts > 0) {
                    ist->next_pts += av_rescale_q(duration_pts, stream->time_base, AV_TIME_BASE_Q);
                } else {
                    ist->next_pts += duration_dts;
                }
            } else
                return 0;
        } else {
            AVRational decoded_frame_tb;
            if (got_output) {
                ist->next_pts += ((int64_t)AV_TIME_BASE * decoded_frm_->nb_samples) /
                                 audio_decode_ctx_->sample_rate;
                ist->next_dts += ((int64_t)AV_TIME_BASE * decoded_frm_->nb_samples) /
                                 audio_decode_ctx_->sample_rate;

                if (stream && decoded_frm_->pts != AV_NOPTS_VALUE) {
                    decoded_frame_tb   = audio_stream_->time_base;
                } else if (stream && pkt && pkt->pts != AV_NOPTS_VALUE) {
                    decoded_frm_->pts = pkt->pts;
                    decoded_frame_tb   = audio_stream_->time_base;
                } else {
                    decoded_frm_->pts = ist->dts;
                    decoded_frame_tb   = AV_TIME_BASE_Q;
                }
                if (decoded_frm_->pts != AV_NOPTS_VALUE)
                    decoded_frm_->pts = av_rescale_delta(decoded_frame_tb, decoded_frm_->pts,
                                                         (AVRational){1, audio_decode_ctx_->sample_rate},
                                                         decoded_frm_->nb_samples,
                                                         &ist->filter_in_rescale_delta_last,
                                                         (AVRational){1, audio_decode_ctx_->sample_rate});
            } else
                return 0;
        }

        frame = av_frame_clone(decoded_frm_);
        if (!frame) {
            BMFLOG_NODE(BMF_ERROR, node_id_) << "Could not allocate frame";
            return -1;
        }
        std::vector<AVFrame *> filter_frames;
        if (!fg_inited_[index]) {
            int ret;
            if ((ret = init_filtergraph(index, frame)) != 0)
                return ret;
            fg_inited_[index] = true;
        }

        if (filter_graph_[index]) {
            int ret = filter_graph_[index]->get_filter_frame(frame, 0, 0, filter_frames);
            if (durations_.size() > 0) {
                if (ret == AVERROR_EOF && dur_end_[index] == false) {
                    //a group of duration trimed finished, switch to another if exist
                    dur_end_[index] = true;
                    if (((task.get_outputs().find(0) != task.get_outputs().end() ? dur_end_[0] : true) &&
                         (task.get_outputs().find(1) != task.get_outputs().end() ? dur_end_[1] : true))) {
                        dur_end_[0] = dur_end_[1] = false;
                        idx_dur_ += 2;
                        if (idx_dur_ >= durations_.size()) {
                            index == 0 ? video_end_ = true : audio_end_ = true;
                            Packet packet = Packet::generate_eof_packet();
                            assert(packet.timestamp_ == BMF_EOF);
                            if (task.get_outputs().find(index) != task.get_outputs().end() && file_list_.size() == 0)
                                task.get_outputs()[index]->push(packet);
                        } else {
                            int64_t timestamp = (int64_t)(durations_[idx_dur_] * AV_TIME_BASE);
                            int s_ret = avformat_seek_file(input_fmt_ctx_, -1, INT64_MIN, timestamp, timestamp, 0);
                            if (s_ret < 0) {
                                BMFLOG_NODE(BMF_ERROR, node_id_) << input_path_.c_str()
                                                                 << "could not seek to position "
                                                                 << (double)(timestamp / AV_TIME_BASE);
                                return s_ret;
                            }
                            //clean the graph and make it to be reinit
                            for (int i = 0; i < 2; i++) {
                                if (filter_graph_[i]) {
                                    delete filter_graph_[i];
                                    fg_inited_[i] = false;
                                }
                            }
                        }
                    }
                }
            }
            if (ret != 0 && ret != AVERROR_EOF) {
                std::string err_msg = "Failed to inject frame into filter network, in decoder";
                BMF_Error(BMF_TranscodeFatalError, err_msg.c_str());
            }
            if (frame)
                av_frame_free(&frame);
        } else
            filter_frames.push_back(frame);

        if (index == 0) {
            if (has_input_ && !push_raw_stream_) { //Special case: for server mode, the frame in different file will be continued
                if (frame->pts == AV_NOPTS_VALUE || frame->pts == last_pts_) {
                    AVRational default_tb = (AVRational){1, fps_};
                    last_pts_ = frame->pts;
                    frame->pts = curr_pts_ + av_rescale_q(1, default_tb, video_stream_->time_base);
                    curr_pts_ = frame->pts;
                } else
                    last_pts_ = frame->pts;
            }

            if (input_timestamp_queue_.size() > 0) {
                frame->pts = input_timestamp_queue_.front();
                input_timestamp_queue_.pop();
            }

            for (int i = 0; i < filter_frames.size(); i++) {
                std::vector<AVFrame*> output_extract_frames;
                extract_frames(filter_frames[i],output_extract_frames);
                for(int j = 0; j < output_extract_frames.size(); j++)
                {
                    Packet packet = generate_video_packet(output_extract_frames[j]);
                    if (task.get_outputs().find(index) != task.get_outputs().end())
                        task.get_outputs()[index]->push(packet);
                        ist->frame_number++;
                }
            }
        } else {
            for (int i = 0; i < filter_frames.size(); i++) {                
                Packet packet = generate_audio_packet(filter_frames[i]);
                if (task.get_outputs().find(index) != task.get_outputs().end())
                    task.get_outputs()[index]->push(packet);
                    ist->frame_number++;
            }
        }

    } else {
        std::vector<AVFrame *> filter_frames;
        if (filter_graph_[index]) {
            int ret = filter_graph_[index]->get_filter_frame(NULL, 0, 0, filter_frames);
            if (ret != 0 && ret != AVERROR_EOF) {
                std::string err_msg = "Failed to inject frame into filter network, in decoder";
                BMF_Error(BMF_TranscodeFatalError, err_msg.c_str());
            }
        }
        for (int i = 0; i < filter_frames.size(); i++) {
            std::vector<AVFrame*> output_extract_frames;
            if (index == 0) {
                extract_frames(filter_frames[i],output_extract_frames);
                for (int j =0; j<output_extract_frames.size(); j++)
                {
                    auto packet = generate_video_packet(output_extract_frames[j]);
                    if (task.get_outputs().find(index) != task.get_outputs().end())
                            task.get_outputs()[index]->push(packet);
                            ist->frame_number++;
                }
            }
            else {
                auto packet = generate_audio_packet(filter_frames[i]);
                if (task.get_outputs().find(index) != task.get_outputs().end())
                        task.get_outputs()[index]->push(packet);
                        ist->frame_number++;
            }
        }

        Packet packet = Packet::generate_eof_packet();
        assert(packet.timestamp() == BMF_EOF);
        if (task.get_outputs().find(index) != task.get_outputs().end() && file_list_.size() == 0)
            task.get_outputs()[index]->push(packet);
    }
    push_data_flag_ = true;
    return 0;
}

int CFFDecoder::pkt_ts(AVPacket *pkt, int index) {
    AVStream *stream = (index == 0) ? video_stream_ : audio_stream_;
    InputStream *ist = &ist_[index];
    int64_t duration;
    int64_t pkt_dts;

    if (!pkt || (pkt && pkt->size == 0))
        return 0;

    if (!ist->wrap_correction_done && input_fmt_ctx_->start_time != AV_NOPTS_VALUE &&
        stream->pts_wrap_bits < 64) {
        int64_t stime, stime2;
        if (ist->next_dts == AV_NOPTS_VALUE
            && ts_offset_ == -input_fmt_ctx_->start_time
            && (input_fmt_ctx_->iformat->flags & AVFMT_TS_DISCONT)) {
            int64_t new_start_time = INT64_MAX;
            for (int i = 0; i < input_fmt_ctx_->nb_streams; i++) {
                AVStream *st = input_fmt_ctx_->streams[i];
                if(st->discard == AVDISCARD_ALL || st->start_time == AV_NOPTS_VALUE)
                    continue;
                new_start_time = FFMIN(new_start_time,
                                       av_rescale_q(st->start_time, st->time_base, AV_TIME_BASE_Q));
            }
            if (new_start_time > input_fmt_ctx_->start_time) {
                BMFLOG_NODE(BMF_INFO, node_id_) << "Correcting start time by " 
                                                << new_start_time - input_fmt_ctx_->start_time;
                ts_offset_ = -new_start_time;
                if (end_time_ > 0) {
                    end_video_time_ = av_rescale_q((end_time_ + ts_offset_), AV_TIME_BASE_Q,
                                                   input_fmt_ctx_->streams[video_stream_index_]->time_base);
                    end_audio_time_ = av_rescale_q((end_time_ + ts_offset_), AV_TIME_BASE_Q,
                                                   input_fmt_ctx_->streams[audio_stream_index_]->time_base);
                }
            }
        }

        stime = av_rescale_q(input_fmt_ctx_->start_time, AV_TIME_BASE_Q, stream->time_base);
        stime2= stime + (1ULL<<stream->pts_wrap_bits);
        ist->wrap_correction_done = 1;

        if (stime2 > stime && pkt->dts != AV_NOPTS_VALUE && pkt->dts > stime + (1LL<<(stream->pts_wrap_bits-1))) {
            pkt->dts -= 1ULL<<stream->pts_wrap_bits;
            ist->wrap_correction_done = 0;
        }
        if (stime2 > stime && pkt->pts != AV_NOPTS_VALUE && pkt->pts > stime + (1LL<<(stream->pts_wrap_bits-1))) {
            pkt->pts -= 1ULL<<stream->pts_wrap_bits;
            ist->wrap_correction_done = 0;
        }
    }

    if (pkt->dts != AV_NOPTS_VALUE)
        pkt->dts += av_rescale_q(ts_offset_, AV_TIME_BASE_Q, stream->time_base);
    if (pkt->pts != AV_NOPTS_VALUE)
        pkt->pts += av_rescale_q(ts_offset_, AV_TIME_BASE_Q, stream->time_base);

    //if (pkt->pts != AV_NOPTS_VALUE)
    //    pkt->pts *= ist->ts_scale;
    //if (pkt->dts != AV_NOPTS_VALUE)
    //    pkt->dts *= ist->ts_scale;

    float dts_delta_threshold = 10;
    pkt_dts = av_rescale_q_rnd(pkt->dts, stream->time_base,
                                AV_TIME_BASE_Q, (enum AVRounding)(AV_ROUND_NEAR_INF|AV_ROUND_PASS_MINMAX));
    if (pkt_dts != AV_NOPTS_VALUE &&
        ist->next_dts == AV_NOPTS_VALUE &&
        //!copy_ts &&
        (input_fmt_ctx_->iformat->flags & AVFMT_TS_DISCONT) &&
        last_ts_ != AV_NOPTS_VALUE
        //!force_dts_monotonicity
    ) {
        int64_t delta   = pkt_dts - last_ts_;
        if (delta < -1LL*dts_delta_threshold*AV_TIME_BASE ||
            delta >  1LL*dts_delta_threshold*AV_TIME_BASE) {
            ts_offset_ -= delta;
            av_log(NULL, AV_LOG_DEBUG,
                   "Inter stream timestamp discontinuity %" PRId64 ", new offset= %" PRId64 "\n",
                   delta, ts_offset_);
            pkt->dts -= av_rescale_q(delta, AV_TIME_BASE_Q, stream->time_base);
            if (pkt->pts != AV_NOPTS_VALUE)
                pkt->pts -= av_rescale_q(delta, AV_TIME_BASE_Q, stream->time_base);
        }
    }

    //duration = av_rescale_q(duration_, ifile->time_base, stream->time_base);  //in loop case
    duration = 0;
    if (pkt->pts != AV_NOPTS_VALUE) {
        pkt->pts += duration;
        ist->max_pts = FFMAX(pkt->pts, ist->max_pts);
        ist->min_pts = FFMIN(pkt->pts, ist->min_pts);
    }
    if (pkt->dts != AV_NOPTS_VALUE)
        pkt->dts += duration;

    pkt_dts = av_rescale_q_rnd(pkt->dts, stream->time_base,
                                AV_TIME_BASE_Q, (enum AVRounding)(AV_ROUND_NEAR_INF|AV_ROUND_PASS_MINMAX));
    if (pkt_dts != AV_NOPTS_VALUE && ist->next_dts != AV_NOPTS_VALUE
        //!copy_ts &&
        //!force_dts_monotonicity
    ) {
        int64_t delta   = pkt_dts - ist->next_dts;
        float dts_error_threshold = 3600*30;
        if (input_fmt_ctx_->iformat->flags & AVFMT_TS_DISCONT) {
            if (delta < -1LL*dts_delta_threshold*AV_TIME_BASE ||
                delta >  1LL*dts_delta_threshold*AV_TIME_BASE ||
                pkt_dts + AV_TIME_BASE/10 < FFMAX(ist->pts, ist->dts)) {
                ts_offset_ -= delta;
                av_log(NULL, AV_LOG_DEBUG,
                       "timestamp discontinuity %" PRId64 ", new offset= %" PRId64 "\n",
                       delta, ts_offset_);
                pkt->dts -= av_rescale_q(delta, AV_TIME_BASE_Q, stream->time_base);
                if (pkt->pts != AV_NOPTS_VALUE)
                    pkt->pts -= av_rescale_q(delta, AV_TIME_BASE_Q, stream->time_base);
            }
        } else {
            if ( delta < -1LL*dts_error_threshold*AV_TIME_BASE ||
                 delta >  1LL*dts_error_threshold*AV_TIME_BASE) {
                av_log(NULL, AV_LOG_WARNING, "DTS %" PRId64 ", next:%" PRId64 " st:%d invalid dropping\n", pkt->dts, ist->next_dts, pkt->stream_index);
                pkt->dts = AV_NOPTS_VALUE;
            }
            if (pkt->pts != AV_NOPTS_VALUE) {
                int64_t pkt_pts = av_rescale_q(pkt->pts, stream->time_base, AV_TIME_BASE_Q);
                delta   = pkt_pts - ist->next_dts;
                if ( delta < -1LL*dts_error_threshold*AV_TIME_BASE ||
                     delta >  1LL*dts_error_threshold*AV_TIME_BASE) {
                    av_log(NULL, AV_LOG_WARNING, "PTS %" PRId64 ", next:%" PRId64 " invalid dropping st:%d\n",
                           pkt->pts, ist->next_dts, pkt->stream_index);
                    pkt->pts = AV_NOPTS_VALUE;
                }
            }
        }
    }

    if (pkt->dts != AV_NOPTS_VALUE)
        last_ts_ = av_rescale_q(pkt->dts, stream->time_base, AV_TIME_BASE_Q);

    //if (force_dts_monotonicity &&
    //    (pkt->pts != AV_NOPTS_VALUE || pkt->dts != AV_NOPTS_VALUE) &&
    //    (ist->dec_ctx->codec_type == AVMEDIA_TYPE_VIDEO || ist->dec_ctx->codec_type == AVMEDIA_TYPE_AUDIO)
    //)
    return 0;
}

int CFFDecoder::decode_send_packet(Task &task, AVPacket *pkt, int *got_frame) {
    int ret = 0;
    int decoded = 0;
    *got_frame = 0;
    int index = pkt->stream_index == video_stream_index_ ? 0 : 1;
    InputStream *ist = &ist_[index];
    AVStream *stream = (index == 0) ? video_stream_ : audio_stream_;
    AVCodecContext *dec_ctx = (index == 0) ? video_decode_ctx_ : audio_decode_ctx_;

    if (!push_raw_stream_)
        pkt_ts(pkt, index);

    if (!ist->saw_first_ts) {
        ist->dts = (stream && stream->avg_frame_rate.num) ? - dec_ctx->has_b_frames * AV_TIME_BASE / av_q2d(stream->avg_frame_rate) : 0;
        ist->pts = 0;
        if (stream && pkt && pkt->pts != AV_NOPTS_VALUE && !ist->decoding_needed) {
            ist->dts += av_rescale_q(pkt->pts, stream->time_base, AV_TIME_BASE_Q);
            ist->pts = ist->dts; //unused but better to set it to a value thats not totally wrong
        }
        ist->saw_first_ts = 1;
    }

    if (ist->next_dts == AV_NOPTS_VALUE)
        ist->next_dts = ist->dts;
    if (ist->next_pts == AV_NOPTS_VALUE)
        ist->next_pts = ist->pts;

    if (stream && pkt && pkt->size != 0 && pkt->dts != AV_NOPTS_VALUE) {
        ist->next_dts = ist->dts = av_rescale_q(pkt->dts, stream->time_base, AV_TIME_BASE_Q);
        if (dec_ctx->codec_type != AVMEDIA_TYPE_VIDEO || !ist->decoding_needed)
            ist->next_pts = ist->pts = ist->dts;
    }

    if (pkt && !ist->decoding_needed) {
        if (pkt->size == 0) {
            Packet packet = Packet::generate_eof_packet();
            assert(packet.timestamp() == BMF_EOF);
            if (task.get_outputs().find(index) != task.get_outputs().end() && file_list_.size() == 0){
                task.get_outputs()[index]->push(packet);
                if (index == 0)
                    video_end_ = true;
                else
                    audio_end_ = true;
            }
            return AVERROR_EOF;
        }

        AVRational frame_rate = av_guess_frame_rate(input_fmt_ctx_, stream, NULL);
        ist->dts = ist->next_dts;
        switch (dec_ctx->codec_type) {
        case AVMEDIA_TYPE_AUDIO:
            assert(pkt->duration >= 0);
            if (dec_ctx->sample_rate) {
                ist->next_dts += ((int64_t)AV_TIME_BASE * dec_ctx->frame_size) /
                                  dec_ctx->sample_rate;
            } else {
                ist->next_dts += av_rescale_q(pkt->duration, stream->time_base, AV_TIME_BASE_Q);
            }
            break;
        case AVMEDIA_TYPE_VIDEO:
            if (frame_rate.num) {
                AVRational time_base_q = AV_TIME_BASE_Q;
                int64_t next_dts = av_rescale_q(ist->next_dts, time_base_q, av_inv_q(frame_rate));
                ist->next_dts = av_rescale_q(next_dts + 1, av_inv_q(frame_rate), time_base_q);
            } else if (pkt->duration) {
                ist->next_dts += av_rescale_q(pkt->duration, stream->time_base, AV_TIME_BASE_Q);
            } else if(dec_ctx->framerate.num != 0) {
                int ticks= av_stream_get_parser(stream) ? av_stream_get_parser(stream)->repeat_pict + 1 : dec_ctx->ticks_per_frame;
                ist->next_dts += ((int64_t)AV_TIME_BASE *
                                  dec_ctx->framerate.den * ticks) /
                                  dec_ctx->framerate.num / dec_ctx->ticks_per_frame;
            }
            break;
        }
        ist->pts = ist->dts;
        ist->next_pts = ist->next_dts;

        if (pkt->dts == AV_NOPTS_VALUE)
            pkt->dts = av_rescale_q(ist->dts, AV_TIME_BASE_Q, stream->time_base);

        if (!ist->codecpar_sended) {
            auto input_stream = std::make_shared<AVStream>();
            *input_stream = *stream;
            input_stream->codecpar = avcodec_parameters_alloc();
            avcodec_parameters_copy(input_stream->codecpar, stream->codecpar);
            auto packet = Packet(input_stream);
            if (task.get_outputs().find(index) != task.get_outputs().end())
                task.get_outputs()[index]->push(packet);
            ist->codecpar_sended = true;
        }
        auto av_packet = ffmpeg::to_bmf_av_packet(pkt, true);
        av_packet.set_pts(pkt->pts);
        av_packet.set_time_base({stream->time_base.num, stream->time_base.den});

        auto packet = Packet(av_packet);
        packet.set_timestamp(pkt->pts * av_q2d(stream->time_base) * 1000000);
        if (task.get_outputs().find(index) != task.get_outputs().end())
            task.get_outputs()[index]->push(packet);

        push_data_flag_ = true;
        return pkt->size;
    }

    bool repeat = false;
    AVPacket *in_pkt = pkt;
    int stream_index = pkt->stream_index;
    AVCodecContext *avctx = (pkt->stream_index == video_stream_index_) ? video_decode_ctx_ :
                                                                          audio_decode_ctx_;
    do {
        ist->pts = ist->next_pts;
        ist->dts = ist->next_dts;

        int64_t dts = AV_NOPTS_VALUE;
        if (index == 0 && stream && ist->dts != AV_NOPTS_VALUE)
            dts = av_rescale_q(ist->dts, AV_TIME_BASE_Q, stream->time_base);

        // The old code used to set dts on the drain packet, which does not work
        // with the new API anymore.
        if (index == 0 && pkt && pkt->size == 0) { // eof
            void *new_buff = av_realloc_array(ist->dts_buffer,
                                              ist->nb_dts_buffer + 1, sizeof(ist->dts_buffer[0]));
            if (!new_buff)
                return AVERROR(ENOMEM);
            ist->dts_buffer = (int64_t *)new_buff;
            ist->dts_buffer[ist->nb_dts_buffer++] = dts;
        }

        if (!repeat) {
            if (index == 0 && pkt && pkt->size != 0)
                pkt->dts = dts; // "ffmpeg.c probably shouldn't do this", but actually it influence

            ret = avcodec_send_packet(avctx, in_pkt);
            if (ret < 0 && ret != AVERROR_EOF ) {//&& ret != AVERROR(EAGAIN)) {
                std::string tmp = (in_pkt->stream_index == video_stream_index_) ? "video" : "audio";
                std::string msg = error_msg(ret);
                BMFLOG_NODE(BMF_ERROR, node_id_) << "Error decoding " << tmp << ", " << msg;
                decode_error_[1]++;
                return ret;
            }
        }

        *got_frame = 0;
        if (decoded_frm_ == NULL) {
            decoded_frm_ = av_frame_alloc();
        }
        ret = avcodec_receive_frame(avctx, decoded_frm_);
        if (ret < 0 && ret != AVERROR(EAGAIN)) {
            if (ret != AVERROR_EOF)
                BMFLOG_NODE(BMF_ERROR, node_id_) << "Error receiving frame";
            //else //loop case
            //    avcodec_flush_buffers(avctx);
            decode_error_[1]++;
            return ret;
        }
        if (ret >= 0) {
            //TODO need to judge the hardware data is from hwaccel or hardware decode
            if (decoded_frm_->hw_frames_ctx) {
                copy_simple_frame(decoded_frm_);
            }
            if (avctx == audio_decode_ctx_)
                ist->sample_decoded += decoded_frm_->nb_samples;
            ist->frame_decoded++;
            *got_frame = 1;
            decode_error_[0]++;
        }

        if (stream_index == video_stream_index_ && !video_end_) {
            // The following line may be required in some cases where there is no parser
            // or the parser does not has_b_frames correctly
            if (video_stream_ && video_stream_->codecpar->video_delay < video_decode_ctx_->has_b_frames) {
                if (video_decode_ctx_->codec_id == AV_CODEC_ID_H264) {
                    video_stream_->codecpar->video_delay = video_decode_ctx_->has_b_frames;
                } else
                    BMFLOG_NODE(BMF_INFO, node_id_) << "video_delay is larger in decoder than demuxer "
                              << video_decode_ctx_->has_b_frames << " > "
                              << video_stream_->codecpar->video_delay;
            }

            if (*got_frame) {
                if (end_video_time_ < decoded_frm_->pts) {
                    if (end_video_time_ < 0)
                        BMFLOG_NODE(BMF_ERROR, node_id_) 
                            << "Error of end time in video, which is shorter than the start offset";
                    handle_output_data(task, 0, in_pkt, true, repeat, *got_frame);
                    video_end_ = true;
                    return 0;
                }
                if (video_stream_ && video_stream_->sample_aspect_ratio.num)
                    decoded_frm_->sample_aspect_ratio = video_stream_->sample_aspect_ratio;
            }
            handle_output_data(task, 0, in_pkt, false, repeat, *got_frame);
        } else if (stream_index == audio_stream_index_ && !audio_end_) {
            if (in_pkt)
                decoded += FFMIN(ret, in_pkt->size);

            if (*got_frame) {
                if (repeat)
                    in_pkt = NULL;
                if (end_audio_time_ < decoded_frm_->pts) {
                    if (end_audio_time_ < 0)
                        BMFLOG_NODE(BMF_INFO, node_id_)
                            << "Error of end time in audio, which is shorter than the start offset";
                    handle_output_data(task, 1, in_pkt, true, repeat, *got_frame);
                    audio_end_ = true;
                    return 0;
                }
            }
            handle_output_data(task, 1, in_pkt, false, repeat, *got_frame);
        }

        // If we use frame reference counting, we own the data and need
        // to de-reference it when we don't use it anymore
        if (*got_frame && refcount_)
            av_frame_unref(decoded_frm_);

        repeat = true;
    } while(*got_frame);

    return decoded;
}

int CFFDecoder::flush(Task &task) {
    AVPacket fpkt;
    int got_frame;
    av_init_packet(&fpkt);
    int ret;

    if (video_stream_index_ != -1) {
        fpkt.stream_index = video_stream_index_;
        while (1) {
            fpkt.data = NULL;
            fpkt.size = 0;
            if (check_valid_packet(&fpkt, task)) { //output q may not exist and won't process
                ret = decode_send_packet(task, &fpkt, &got_frame);
                if (ret < 0) {
                    if (ret == AVERROR_EOF)
                        break;
                    if (ret != AVERROR(EAGAIN)) {
                        std::string msg = error_msg(ret);
                        BMFLOG_NODE(BMF_ERROR, node_id_) << "flush decode video error: " << msg;
                    }
                }
            } else
                break;
        }
    }

    if (audio_stream_index_ != -1) {
        fpkt.stream_index = audio_stream_index_;
        while (1) {
            fpkt.data = NULL;
            fpkt.size = 0;
            if (check_valid_packet(&fpkt, task)) {
                ret = decode_send_packet(task, &fpkt, &got_frame);
                if (ret < 0) {
                    if (ret == AVERROR_EOF)
                        break;
                    if (ret != AVERROR(EAGAIN)) {
                        std::string msg = error_msg(ret);
                        BMFLOG_NODE(BMF_ERROR, node_id_) << "flush decode audio error" << msg;
                        break;
                    }
                }
            } else
                break;
        }
    }

    BMFLOG_NODE(BMF_INFO, node_id_) << "decode flushing";
    if (!audio_end_) {
        handle_output_data(task, 1, NULL, true, false, got_frame);
        audio_end_ = true;
    }
    if (!video_end_) {
        handle_output_data(task, 0, NULL, true, false, got_frame);
        video_end_ = true;
        next_file_ = true;
    }

    return 0;
}

int CFFDecoder::close() {
    std::lock_guard<std::mutex> lock(mutex_);
    clean();
    if ((decode_error_[0] + decode_error_[1]) * max_error_rate_ < decode_error_[1]) {
        std::string err_msg = "decoded: " + std::to_string(decode_error_[0]) + " , failed to decode: "
                              + std::to_string(decode_error_[1]) + ".";
        BMF_Error(BMF_TranscodeError, err_msg.c_str());
    }
    return 0;
}

int CFFDecoder::clean() {
    if (avio_ctx_) {
        av_freep(&avio_ctx_->buffer);
        av_freep(&avio_ctx_);
    }
    if (decoded_frm_) {
        av_frame_free(&decoded_frm_);
        decoded_frm_ = NULL;
    }
    if (video_decode_ctx_) {
        avcodec_free_context(&video_decode_ctx_);
        video_decode_ctx_ = NULL;
    }
    if (audio_decode_ctx_) {
        avcodec_free_context(&audio_decode_ctx_);
        audio_decode_ctx_ = NULL;
    }
    if (input_fmt_ctx_) {
        avformat_close_input(&input_fmt_ctx_);
        input_fmt_ctx_ = NULL;
    }
    if (ist_[0].dts_buffer)
        av_freep(&ist_[0].dts_buffer);
    for (int i = 0; i < 2; i++) {
        if (filter_graph_[i]) {
            delete filter_graph_[i];
            filter_graph_[i] = NULL;
        }
    }
    if (dec_opts_)
        av_dict_free(&dec_opts_);

    fg_inited_[0] = false;
    fg_inited_[1] = false;
    start_decode_flag_ = false;
    handle_input_av_packet_flag_ = false;
    packets_handle_all_ = false;
    valid_packet_flag_ = false;
    stream_frame_number_ = 0;
    while(!bmf_av_packet_queue_.empty())
    {
        bmf_av_packet_queue_.pop();
    }
    return 0;
}

int CFFDecoder::read_packet(uint8_t *buf, int buf_size) {
    int valid_size = 0;
    if (bmf_av_packet_queue_.empty()) {
        std::unique_lock<std::mutex> lk(process_mutex_);
        packets_handle_all_ = true;
        valid_packet_flag_ = false;
        process_var_.notify_one();
        packet_ready_.wait(lk,[this]{return this->valid_packet_flag_;});

    }
    while (not bmf_av_packet_queue_.empty()) {
        BMFAVPacket packet = bmf_av_packet_queue_.front();
        if (packet.pts() == BMF_EOF) {
            if (valid_size == 0) {
                // bmf_av_packet_queue_.pop();
                return AVERROR_EOF;
            }
            return valid_size;
        }
        int current_packet_valid_size = packet.nbytes() - current_packet_loc_;
        int need_size = buf_size - valid_size;
        int got_size = FFMIN(need_size, current_packet_valid_size);
        memcpy(buf + valid_size, (unsigned char *) (packet.data_ptr()) + current_packet_loc_, got_size);
        valid_size += got_size;

        if (current_packet_loc_ + got_size >= packet.nbytes()) {
            bmf_av_packet_queue_.pop();
            current_packet_loc_ = 0;
        } else {
            current_packet_loc_ = current_packet_loc_ + got_size;
        }
        if (valid_size >= buf_size) {
            break;
        }
    }
    return valid_size;

}

int read_packet_(void *opaque, uint8_t *buf, int buf_size) {
    return ((CFFDecoder *) opaque)->read_packet(buf, buf_size);
}

int CFFDecoder::init_av_codec() {
    //AVDictionary *opts = NULL;
    input_fmt_ctx_ = NULL;
    video_time_base_string_ = "";
    video_end_ = false;
    audio_end_ = false;
    video_stream_index_ = -1;
    audio_stream_index_ = -1;
    init_input(dec_opts_);
    return 0;
}

bool CFFDecoder::check_valid_packet(AVPacket *pkt, Task &task) {
    if (pkt->stream_index == video_stream_index_ && !video_end_ && task.get_outputs().count(0) > 0) {
        return true;
    }
    if (pkt->stream_index == audio_stream_index_ && !audio_end_ && task.get_outputs().count(1) > 0) {
        return true;
    }
    return false;
}

int CFFDecoder::init_packet_av_codec() {
    unsigned char *avio_ctx_buffer;
    size_t avio_ctx_buffer_size = 1024;
    input_fmt_ctx_ = avformat_alloc_context();

    avio_ctx_buffer = (unsigned char *) av_malloc(avio_ctx_buffer_size);

    avio_ctx_ = avio_alloc_context(avio_ctx_buffer, avio_ctx_buffer_size, 0, (void *) this, read_packet_,
                                   NULL, NULL);
    input_fmt_ctx_->pb = avio_ctx_;
    input_fmt_ctx_->flags = AVFMT_FLAG_CUSTOM_IO;
    video_end_ = false;
    audio_end_ = false;
    start_time_ = AV_NOPTS_VALUE;
    video_stream_index_ = -1;
    audio_stream_index_ = -1;
    video_time_base_string_ = "";
    AVDictionary *opts = NULL;
    init_input(opts);

    return 0;
}


int CFFDecoder::start_decode(std::vector<int> input_index, std::vector<int> output_index) {

    start_decode_flag_ = true;
    int ret =0;
    task_ = Task(node_id_, input_index, output_index);
    init_packet_av_codec();
    if (!video_stream_ && video_end_ != true) {
        handle_output_data(task_, 0, NULL, true, false, 0);
        video_end_ = true;
    }
    if (!audio_stream_ && audio_end_ != true) {
        handle_output_data(task_, 1, NULL, true, false, 0);
        audio_end_ = true;
    }

    AVPacket pkt;
    push_data_flag_ = false;
    int got_frame = 0;
    while (!(video_end_ && audio_end_)) {
        av_init_packet(&pkt);
        ret = av_read_frame(input_fmt_ctx_, &pkt);

        if (ret < 0 ) {
            flush(task_);
            if (file_list_.size() == 0) {
                task_.set_timestamp(DONE);
                task_done_ = true;
            }
            break;
        }
        if (ret >= 0 && check_valid_packet(&pkt, task_)) {
            ret = decode_send_packet(task_, &pkt, &got_frame);
        }
        av_packet_unref(&pkt);
        if (ret == AVERROR_EOF || (video_end_ && audio_end_))  {
            flush(task_);
            if (file_list_.size() == 0){
                task_.set_timestamp(DONE);
                task_done_ = true;
            }
            break;
        }
    }

    if (task_done_)
        task_.set_timestamp(DONE);
    packets_handle_all_ = true;
    valid_packet_flag_ = false;
    process_var_.notify_one();
    return 0;
}

int64_t CFFDecoder::get_start_time() {
    int64_t duration_ = 10000000;
    // int64_t start_time = 0;
    int64_t start_time = -ts_offset_;
    int64_t adjust_time = 0;
    if (first_video_start_time_ == -1) {
        first_video_start_time_ = start_time;
    }
    if (last_output_pts_ == 0)
    {
        return start_time;
    }
    if (start_time < last_output_pts_ - overlap_time_) {
        adjust_time = last_output_pts_ + cut_off_interval_;
    }
    else if(start_time <= last_output_pts_) {
        adjust_time = last_output_pts_ + cut_off_interval_;
    }
    else if(start_time < last_output_pts_ + cut_off_time_) {
        //the same with normal packet
        adjust_time = start_time;
    } else {
        // correct the timestamp and add
        adjust_time = last_output_pts_ + cut_off_interval_;
    }
    return adjust_time;
}

int CFFDecoder::process_task_output_packet(int index, Packet &packet) {
    int64_t adjust_pts = 0;
    if (first_handle_) {
        adjust_pts_ = get_start_time();
        first_handle_ = false;
    }
    adjust_pts = adjust_pts_;
    if(packet.is<std::shared_ptr<AVStream>>()){
        if (!stream_copy_av_stream_flag_[index]){
            stream_copy_av_stream_flag_[index] = true;
            return 0;
        } else {
            return -1;
        }
    }

    if (adjust_pts + packet.timestamp() <= last_output_pts_){
        return -1;
    }

    if (adjust_pts + packet.timestamp() > temp_last_output_pts_){
        temp_last_output_pts_ = adjust_pts + packet.timestamp();
    }
    packet.set_timestamp(adjust_pts + packet.timestamp());
    if(packet.is<VideoFrame>()){
        auto& video_frame = packet.get<VideoFrame>();
        auto frame = const_cast<AVFrame*>(video_frame.private_get<AVFrame>());
        frame->pts = av_rescale_q(adjust_pts - first_video_start_time_, AV_TIME_BASE_Q, video_stream_->time_base) + frame->pts;
        video_frame.set_pts(frame->pts); //sync with avframe
    }
    else if(packet.is<AudioFrame>()){
        auto& audio_frame = packet.get<AudioFrame>();
        auto frame = const_cast<AVFrame*>(audio_frame.private_get<AVFrame>());
        
        Rational tb = audio_frame.time_base();
        AVRational audio_frame_time_base;
        audio_frame_time_base.den = tb.den;
        audio_frame_time_base.num = tb.num;
        frame->pts = av_rescale_q(adjust_pts - first_video_start_time_, AV_TIME_BASE_Q, audio_frame_time_base) + frame->pts; 

        audio_frame.set_pts(frame->pts); //sync with avframe
    }
    else if(packet.is<BMFAVPacket>()){
        auto& bmf_av_packet = packet.get<BMFAVPacket>();
        auto av_packet = const_cast<AVPacket*>(bmf_av_packet.private_get<AVPacket>());
        if (index == 0){
            av_packet->pts = av_rescale_q(adjust_pts - first_video_start_time_, AV_TIME_BASE_Q, video_stream_->time_base) + av_packet->pts;
            av_packet->dts = av_rescale_q(adjust_pts - first_video_start_time_, AV_TIME_BASE_Q, video_stream_->time_base) + av_packet->dts;
        } else {
            av_packet->pts = av_rescale_q(adjust_pts - first_video_start_time_, AV_TIME_BASE_Q, audio_stream_->time_base) + av_packet->pts; 
            av_packet->dts = av_rescale_q(adjust_pts - first_video_start_time_, AV_TIME_BASE_Q, audio_stream_->time_base) + av_packet->dts;
        }

        bmf_av_packet.set_pts(av_packet->pts); //sync with AVPacket
    }
    return 0;
}

int CFFDecoder::mv_task_data(Task &dst_task) {
    std::vector<int> output_indexs = dst_task.get_output_stream_ids();
    
    for (int i =0; i<output_indexs.size(); i++) {
        Packet pkt;
        while(task_.pop_packet_from_out_queue(output_indexs[i], pkt)){
            if (pkt.timestamp() == BMF_EOF) {
                if (input_eof_packet_received_){
                    dst_task.fill_output_packet(output_indexs[i], pkt);
                }
            } else {
                int ret = process_task_output_packet(output_indexs[i], pkt);
                if (ret >= 0) {
                    dst_task.fill_output_packet(output_indexs[i], pkt);
                }
            }
            
        }
    }
    if (task_.timestamp() == DONE ){
        first_handle_ = true;
        if (input_eof_packet_received_) {
            dst_task.set_timestamp(DONE);
        }
        else {
            clean();
            last_output_pts_ = temp_last_output_pts_;
        }
    }
    
    return 0;
}

int CFFDecoder::process_raw_stream_packet(Task &task, BMFAVPacket &bmf_pkt, bool eof) {
    int got_frame;
    if (!video_codec_name_.empty() && !video_decode_ctx_) {
        video_stream_index_ = 0;
        AVCodec *dec = avcodec_find_decoder_by_name(video_codec_name_.c_str());
        if (!dec)
            BMFLOG_NODE(BMF_ERROR, node_id_) << "Codec not found";
        video_decode_ctx_ = avcodec_alloc_context3(dec);
        if (!video_decode_ctx_)
            BMFLOG_NODE(BMF_ERROR, node_id_) << "Context not found";
        video_decode_ctx_->codec_type = AVMEDIA_TYPE_VIDEO;
        AVDictionary *opts = dec_opts_;
        av_dict_set(&opts, "refcounted_frames", "1", 0);
        av_dict_set(&opts, "threads", "auto", 0);
        if (avcodec_open2(video_decode_ctx_, dec, &opts) < 0)
            BMFLOG_NODE(BMF_ERROR, node_id_) << "Could not open codec";

        // Parse time base
        std::vector<int> time_base_comps;
        std::stringstream stbs(video_time_base_string_);
        for (int i; stbs >> i;) {
            time_base_comps.push_back(i);
            if (stbs.peek() == ',')
                stbs.ignore();
        }
        if (time_base_comps.size() != 2)
            throw std::runtime_error("Invalid video time base provided to decoder");
        video_time_base_ = (AVRational){time_base_comps[0], time_base_comps[1]};
    } else if (!audio_codec_name_.empty() && !audio_decode_ctx_) {
        audio_stream_index_ = 0;
        AVCodec *dec = avcodec_find_decoder_by_name(audio_codec_name_.c_str());
        if (!dec)
            BMFLOG_NODE(BMF_ERROR, node_id_) << "Codec not found";
        audio_decode_ctx_ = avcodec_alloc_context3(dec);
        if (!audio_decode_ctx_)
            BMFLOG_NODE(BMF_ERROR, node_id_) << "Context not found";
        audio_decode_ctx_->codec_type = AVMEDIA_TYPE_AUDIO;
        audio_decode_ctx_->channels = push_audio_channels_;
        audio_decode_ctx_->sample_rate = push_audio_sample_rate_;
        if (push_audio_sample_fmt_) audio_decode_ctx_->sample_fmt = static_cast<AVSampleFormat>(push_audio_sample_fmt_);
        if (avcodec_open2(audio_decode_ctx_, dec, NULL) < 0)
            BMFLOG_NODE(BMF_ERROR, node_id_) << "Could not open codec";
    }
    //
    AVPacket *av_packet = nullptr;
    if(bmf_pkt){
        av_packet = ffmpeg::from_bmf_av_packet(bmf_pkt, false);
    }
    else{
        av_packet = av_packet_alloc();
        av_packet->size = 0;
    }

    int decode_len = decode_send_packet(task, av_packet, &got_frame);
    av_packet_unref(av_packet);
    if (decode_len < 0 && decode_len != AVERROR(EAGAIN) && decode_len != AVERROR_EOF
        && !(video_end_ && audio_end_))
        BMFLOG_NODE(BMF_ERROR, node_id_) << "Error of decode raw stream";

    if (eof) {
        flush(task);
        task.set_timestamp(DONE);
    }
    return 0;
}

int CFFDecoder::process_input_bmf_av_packet(Task &task){
    Packet packet;
    if ((task.inputs_queue_.count(0)) <= 0) {
        BMFLOG_NODE(BMF_ERROR, node_id_) << "the av packet should push to stream 0";
        return -1;
    }
    std::unique_lock<std::mutex> lck(process_mutex_);
    while (task.pop_packet_from_input_queue(0, packet)) {
        if (packet.timestamp() == BMF_EOF) {
            input_eof_packet_received_ = true;
            if (!start_decode_flag_) {
                task.fill_output_packet(0,Packet::generate_eof_packet());
                task.fill_output_packet(1,Packet::generate_eof_packet());
                task.set_timestamp(DONE);
            }
            BMFAVPacket bmf_pkt;
            if (push_raw_stream_) {
                process_raw_stream_packet(task, bmf_pkt, true);
                continue;
            }
            bmf_pkt.set_pts((int64_t)BMF_EOF);
            bmf_av_packet_queue_.push(bmf_pkt);
            packets_handle_all_ = false;
            break;
        }
        auto bmf_pkt = packet.get<BMFAVPacket>();
        if (push_raw_stream_) {
            process_raw_stream_packet(task, bmf_pkt, false);
            continue;
        }

        // when the bmf_pkt size is 0, which mean the file is end, and need to be ready to recive the another format file
        if (bmf_pkt.nbytes() == 0)
        {
            BMFAVPacket bmf_pkt;
            bmf_pkt.set_pts(BMF_EOF);
            bmf_av_packet_queue_.push(bmf_pkt);
            packets_handle_all_ = false;
            break;
        }
        bmf_av_packet_queue_.push(bmf_pkt);
        packets_handle_all_ = false;
    }
    if (push_raw_stream_)
        return 0;
    if (!start_decode_flag_) {
        exec_thread_ = std::thread(&CFFDecoder::start_decode, this, task.get_input_stream_ids(),task.get_output_stream_ids());
    }
    valid_packet_flag_ = true;
    packet_ready_.notify_one();
    process_var_.wait(lck,[this]{return this->packets_handle_all_;});
    mv_task_data(task);
    if (task_.timestamp()==DONE){
        exec_thread_.join();
    }
    return 0;
}

int CFFDecoder::process(Task &task) {
    std::lock_guard<std::mutex> lock(mutex_);
    int ret = 0, got_frame;

    if (has_input_) {
        if (task.get_inputs().size() > 0 && task.get_inputs()[0]->size() > 0) {
            Packet packet = task.get_inputs()[0]->front();
            if(packet.is<BMFAVPacket>() || handle_input_av_packet_flag_){
                handle_input_av_packet_flag_ = true;
                return process_input_bmf_av_packet(task);
            }
        }

        for (int index = 0; index < task.get_inputs().size(); index++) {
            Packet packet;
            while (task.pop_packet_from_input_queue(index, packet)) {
                if (packet.timestamp() == BMF_EOF) {
                    continue;
                }
                // mainly used to cowork with sync inputstream manager
                if (packet.timestamp() == BMF_PAUSE) {
                    for (auto output:task.get_outputs())
                        output.second->push(packet);
                    break;
                }

                curr_pts_ = 0;
                last_pts_ = AV_NOPTS_VALUE;
                task_done_ = false;
                next_file_ = true;

                auto json_data = packet.get<JsonParam>();
                json_data.get_object_list("input_path", file_list_);
            }
        }
        if (next_file_ && file_list_.size() > 0) {
            next_file_ = false;
            JsonParam file_info = file_list_.front();
            file_list_.erase(file_list_.begin());
            file_info.get_string("input_path", input_path_);
            init_av_codec();
        }
    }

    if (!input_fmt_ctx_) {
        BMFLOG_NODE(BMF_WARNING, node_id_) << "decoder input_fmt_ctx_ is not ready or might be free";
        return 0;
    }
    if (!video_stream_ && video_end_ != true) {
        handle_output_data(task, 0, NULL, true, false, 0);
        video_end_ = true;
    }
    if (!audio_stream_ && audio_end_ != true) {
        handle_output_data(task, 1, NULL, true, false, 0);
        audio_end_ = true;
    }

    AVPacket pkt;
    push_data_flag_ = false;
    while (!(video_end_ && audio_end_)) {
        av_init_packet(&pkt);
        ret = av_read_frame(input_fmt_ctx_, &pkt);
        if (ret == AVERROR(EAGAIN)) {
            usleep(10000);
            continue;
        }
        if (ret < 0 ) {
            flush(task);
            if (file_list_.size() == 0) {
                task.set_timestamp(DONE);
                task_done_ = true;
            }
            break;
        }

        if (ret >= 0 && check_valid_packet(&pkt, task)) {
            ret = decode_send_packet(task, &pkt, &got_frame);
            if (ret < 0 && ret != AVERROR(EAGAIN) && ret != AVERROR_EOF
                && !(video_end_ && audio_end_))
                break;
        }
        av_packet_unref(&pkt);
        if (ret == AVERROR_EOF || (video_end_ && audio_end_)) {
            flush(task);
            if (file_list_.size() == 0) {
                task.set_timestamp(DONE);
                task_done_ = true;
            }
            break;
        } else if (push_data_flag_) {
            break;
        }
    }

    if (task_done_)
        task.set_timestamp(DONE);
    return PROCESS_OK;
}

REGISTER_MODULE_CLASS(CFFDecoder)
