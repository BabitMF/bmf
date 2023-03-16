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

#include "ffmpeg_encoder.h"
#include "libswresample/swresample.h"
#include <filesystem>
#include <iostream>
#include <sys/types.h>
#include <sys/stat.h>
#include <dirent.h>
#include <cstddef>
#include <bmf/sdk/video_frame.h>
#include <bmf/sdk/audio_frame.h>
#include <bmf/sdk/ffmpeg_helper.h>
#include <bmf/sdk/log_buffer.h>
#include <bmf/sdk/bmf_av_packet.h>
#include <bmf/sdk/exception_factory.h>
#include <bmf/sdk/error_define.h>
USE_BMF_SDK_NS

CFFEncoder::CFFEncoder(int node_id, JsonParam option) {
    node_id_ = node_id;
    input_option_ = option;
    output_path_ = "";
    output_prefix_ = "";
    null_output_ = false;

    /** @addtogroup EncM
     * @{
     * @arg null_output: to make encoder as a null sink in some cases. "null_output": 1
     * @} */
    if (option.has_key("null_output"))
        null_output_ = true;

    /** @addtogroup EncM
     * @{
     * @arg output_path: output file path, exp. out.mp4, which can indicate the output format of the file similiar as ffmpeg.
     * @} */
    if (option.has_key("output_path")) {
        option.get_string("output_path", output_path_);
    } else if (!option.has_key("output_prefix") && !null_output_) {
        BMFLOG_NODE(BMF_ERROR, node_id_) << "No output path";
        return;
    }

    /** @addtogroup EncM
     * @{
     * @arg adjust_pts: will adjust the pts start from 0 when it's enabled
     * @} */
    if (option.has_key("adjust_pts")) {
        int adjust_pts;
        option.get_int("adjust_pts", adjust_pts);
        if (adjust_pts == 1) {
            adjust_pts_flag_ = true;
        }

    }
    if (option.has_key("output_prefix")) {
        option.get_string("output_prefix", output_prefix_);
        if (output_prefix_[output_prefix_.size() - 1] != '/')
            output_prefix_ += '/';
        if(!std::filesystem::is_directory(output_prefix_)){
            mkdir(output_prefix_.c_str(), S_IRWXU);
        }
    }

    srv_cnt_ = 0;
    init();
    return;
}

int CFFEncoder::init() {
    if (b_init_) {
        return 0;
    }
    b_init_ = true;
    reset_flag_ = false;
    output_fmt_ctx_ = NULL;
    oformat_ = "mp4";
    codecs_[0] = NULL;
    codecs_[1] = NULL;
    enc_ctxs_[0] = NULL;
    enc_ctxs_[1] = NULL;
    output_stream_[0] = NULL;
    output_stream_[1] = NULL;
    in_stream_tbs_[0] = (AVRational){-1, -1};
    sws_ctx_ = NULL;
    swr_ctx_ = NULL;
    codec_names_[0] = "libx264";
    codec_names_[1] = "aac";
    streams_idx_[0] = 0;
    streams_idx_[1] = 1;
    width_ = 0;
    height_ = 0;
    last_pts_ = -1;
    recorded_pts_ = -1;
    fps_ = 0;
    n_resample_out_ = 0;
    pix_fmt_ = AV_PIX_FMT_YUV420P;
    num_input_streams_ = 0;
    codec_init_touched_num_ = 0;
    stream_inited_ = false;
    b_stream_eof_[0] = false;
    b_stream_eof_[1] = false;
    b_flushed_ = false;
    b_eof_ = false;
    ost_[0] = ost_[1] = {0};
    ost_[0].last_mux_dts = ost_[1].last_mux_dts = AV_NOPTS_VALUE;
    ost_[0].encoding_needed = ost_[1].encoding_needed = true;
    ost_[0].filter_in_rescale_delta_last = ost_[1].filter_in_rescale_delta_last = AV_NOPTS_VALUE;
    ost_[0].max_frames = ost_[1].max_frames = INT64_MAX;

    /** @addtogroup EncM
     * @{
     * @arg format: similiar as the "-f" in ffmpeg command line to specify the demux/mux format. exp.
     * @code
     * {
           "format": "flv",
           "output_path": rtmp://docker.for.mac.host.internal/rtmplive
     * }
     * @endcode
     * @} */
    oformat_ = "mp4";
    if (output_path_.substr(output_path_.find_last_of('.') + 1) == "m3u8")
        oformat_ = "hls";
    if (input_option_.has_key("format"))
        input_option_.get_string("format", oformat_);

    /** @addtogroup EncM
     * @{
     * @arg output_prefix: specify the output directory path
     * @} */
    srv_cnt_++;
    if (input_option_.has_key("output_prefix")) {
        output_dir_ = output_prefix_ + std::to_string(srv_cnt_);
        if(!std::filesystem::is_directory(output_dir_)){
            mkdir(output_dir_.c_str(), S_IRWXU);
        }
        output_path_ = output_dir_ + "/output." + oformat_;
    }

    /** @addtogroup EncM
     * @{
     * @arg push_output: decide whether to mux the result and where to output the results, available value is 0/1/2. 0: write muxed result to disk, 1: write muxed result to the output queue, 2: write unmuxed result to the output queue.
     * @code
            "push_output": 1
     * @endcode
     * @} */
    if (input_option_.has_key("push_output")) {
        input_option_.get_int("push_output", push_output_);
    }

    /** @addtogroup EncM
     * @{
     * @arg avio_buffer_size: set avio buffer size, when oformat is image2pipe, this paramter is useful, exp.
     * @code
            "avio_buffer_size": 16384
     * @endcode
     * @} */
    if (input_option_.has_key("avio_buffer_size")) {
        int tmp;
        input_option_.get_int("avio_buffer_size", tmp);
        avio_buffer_size_ = tmp;
    } else {
        avio_buffer_size_ = 4 * 4096;
    }

    /** @addtogroup EncM
     * @{
     * @arg mux_params: specify the extra output mux parameters, exp.
     * @code
            "format": "hls",
            "mux_params": {
                "hls_list_size": "0",
                "hls_time": "10",
                "hls_segment_filename": "./file%03d.ts"
            }
     * @endcode
     * @} */
    if (input_option_.has_key("mux_params")) {
        input_option_.get_object("mux_params", mux_params_);
    }

    /** @addtogroup EncM
     * @{
     * @arg video_params: video codec related parameters which similiar as ffmpeg. exp.
     * @code
            "video_params": {
                "codec": "h264",
                "width": 320,
                "height": 240,
                "crf": 23,
                "preset": "veryfast"
            },
     * @endcode
     * @} */
    if (input_option_.has_key("video_params")) {
        input_option_.get_object("video_params", video_params_);
    }

    /** @addtogroup EncM
     * @{
     * @arg metadata: to add user metadata in the outfile
     * @} */
    if (input_option_.has_key("metadata")) {
        input_option_.get_object("metadata", metadata_params_);
    }

    /** @addtogroup EncM
     * @{
     * @arg vframes: set the number of video frames to output
     * @} */
    if (input_option_.has_key("vframes"))
        input_option_.get_long("vframes", ost_[0].max_frames);
    /** @addtogroup EncM
     * @{
     * @arg aframes: set the number of audio frames to output
     * @} */
    if (input_option_.has_key("aframes"))
        input_option_.get_long("aframes", ost_[1].max_frames);
    assert(ost_[0].max_frames >= 0 && ost_[1].max_frames >= 0);
    /** @addtogroup EncM
     * @{
     * @arg min_frames: set the min number of output video frames
     * @} */
    if (input_option_.has_key("min_frames"))
        input_option_.get_long("min_frames", ost_[0].min_frames);

    /** @addtogroup EncM
     * @{
     * @arg codec: param in video_params or audio_params to specify the name of the codec which libavcodec included. exp. "h264", "bytevc1", "jpg", "png", "aac"(audio)
     * @} */
    std::string codec;
    if (video_params_.has_key("codec")) {
        video_params_.get_string("codec", codec);
        video_params_.erase("codec");
        if (codec == "h264") {
            codec_names_[0] = "libx264";
        } else if (codec == "bytevc1") {
            codec_names_[0] = "bytevc1";
        } else if (codec == "jpg") {
            codec_names_[0] = "mjpeg";
            pix_fmt_ = AV_PIX_FMT_YUVJ444P;
        } else if (codec == "png") {
            codec_names_[0] = "png";
            pix_fmt_ = AV_PIX_FMT_RGBA;
        } else {
            codec_names_[0] = codec;
        }
    }
    /** @addtogroup EncM
     * @{
     * @arg width: param in video_params to specify the video width
     * @arg height: param in video_params to specify the video height
     * @arg pix_fmt: param in video_params to specify the input format of raw video
     * @} */
    if (video_params_.has_key("width") && video_params_.has_key("height")) {
        video_params_.get_int("width", width_);
        video_params_.get_int("height", height_);
        video_params_.erase("width");
        video_params_.erase("height");
    }

    if (video_params_.has_key("pix_fmt") ) {
        std::string pix_fmt_str ;
        video_params_.get_string("pix_fmt", pix_fmt_str);
        pix_fmt_ = av_get_pix_fmt(pix_fmt_str.c_str());
        video_params_.erase("pix_fmt");
    }

    /** @addtogroup EncM
     * @{
     * @arg audio_params: audio codec related parameters which similiar as ffmpeg. exp.
     * @code
           "audio_params": {
               "codec": "aac",
               "bit_rate": 128000,
               "sample_rate": 44100,
               "channels": 2
           }
     * @endcode
     * @} */
    if (input_option_.has_key("audio_params")) {
        input_option_.get_object("audio_params", audio_params_);
    }
    if (audio_params_.has_key("codec")) {
        audio_params_.get_string("codec", codec_names_[1]);
        audio_params_.erase("codec");
    }

    /** @addtogroup EncM
     * @{
     * @arg loglevel: without using the logbuffer of builder API, to set the ffmpeg av log level: "quiet","panic","fatal","error","warning","info","verbose","debug","trace"
     * @} */
    if (!LogBuffer::avlog_cb_set())
        av_log_set_level(AV_LOG_VERBOSE);
    if (input_option_.has_key("loglevel")) {
        std::string log_level = "";
        input_option_.get_string("loglevel", log_level);
        if (!LogBuffer::avlog_cb_set()) {
            av_log_set_level(LogBuffer::infer_level(log_level));
            BMFLOG_NODE(BMF_INFO, node_id_) << "encode setting log level to: " << log_level;
        }
    }

    return 0;
}

CFFEncoder::~CFFEncoder() {
    clean();
}

int CFFEncoder::clean() {
    if (!b_init_)
        return 0;
    if (avio_ctx_) {
        av_freep(&avio_ctx_->buffer);
        av_freep(&avio_ctx_);
    }
    if (current_image_buffer_.buf) {
        av_freep(&(current_image_buffer_.buf));
        current_image_buffer_.size = 0;
        current_image_buffer_.room = 0;
    }
    for (int idx = 0; idx <= 1; idx++) {
        if (codecs_[idx]) {
            codecs_[idx] = NULL;
        }
        if (enc_ctxs_[idx]) {
            avcodec_free_context(&enc_ctxs_[idx]);
            enc_ctxs_[idx] = NULL;
        }
        if (ost_[idx].input_stream)
            ost_[idx].input_stream = NULL;
    }
    if (push_output_ == OutputMode::OUTPUT_NOTHING && output_fmt_ctx_ && output_fmt_ctx_->oformat && !(output_fmt_ctx_->oformat->flags & AVFMT_NOFILE))
        avio_closep(&output_fmt_ctx_->pb);

    if (output_fmt_ctx_) {
        avformat_free_context(output_fmt_ctx_);
        output_fmt_ctx_ = NULL;
    }

    if (sws_ctx_) {
        sws_freeContext(sws_ctx_);
        sws_ctx_ = NULL;
    }
    if (swr_ctx_) {
        swr_free(&swr_ctx_);
        swr_ctx_ = NULL;
    }

    return 0;
}

int CFFEncoder::reset() {
    if (reset_flag_)
        return 0;
    flush();
    clean();
    video_sync_ = NULL;
    output_video_filter_graph_ = NULL;
    output_audio_filter_graph_ = NULL;
    reset_flag_ = true;
    b_init_ = false;
    return 0;
}

bool CFFEncoder::check_valid_task(Task& task) {
    for (int index = 0; index < task.get_inputs().size(); index++) {
        if (!task.get_inputs()[index]->empty()) {
            return true;
        }
    }
    return false;
}

int CFFEncoder::handle_output(AVPacket *hpkt, int idx) {
    int ret;
    AVPacket *pkt = hpkt;

    if (idx == 0) {
        if(callback_endpoint_ != NULL) {
            float curr_time = (in_stream_tbs_[0].den > 0 && in_stream_tbs_[0].num > 0) ?
                              float(pkt->pts * in_stream_tbs_[0].num / in_stream_tbs_[0].den) : 0;

            std::string info = "pts: " + std::to_string(curr_time);
            auto para = CBytes::make((uint8_t*)info.c_str(), info.size());
            callback_endpoint_(0, para);
        }
    }

    if (push_output_) {
        current_frame_pts_ = pkt->pts;
        orig_pts_time_ = -1;
        if (orig_pts_time_list_.size() > 0) {
            orig_pts_time_ = orig_pts_time_list_.front();
            orig_pts_time_list_.pop_front();
        }
    }

    AVFormatContext *s = output_fmt_ctx_;
    AVStream *st = output_stream_[idx];
    OutputStream *ost = &ost_[idx];

    AVPacket out_pkt = { 0 };
    if (!ost->encoding_needed)
        if (streamcopy(hpkt, &out_pkt, idx) != 0) {
            BMFLOG_NODE(BMF_ERROR, node_id_) << "stream copy error";
            return -1;
        } else
            pkt = &out_pkt;

    if (!(st->codecpar->codec_type == AVMEDIA_TYPE_VIDEO && ost->encoding_needed)) {
        if (ost->frame_number >= ost->max_frames) {
            av_packet_unref(pkt);
            return 0;
        }
        ost->frame_number++;
    }

    //if (!of->header_written) {
    //    AVPacket tmp_pkt = {0};
    //    /* the muxer is not initialized yet, buffer the packet */
    //    if (!av_fifo_space(ost->muxing_queue)) {
    //        int new_size = FFMIN(2 * av_fifo_size(ost->muxing_queue),
    //                             ost->max_muxing_queue_size);
    //        if (new_size <= av_fifo_size(ost->muxing_queue)) {
    //            av_log(NULL, AV_LOG_ERROR,
    //                   "Too many packets buffered for output stream %d:%d.\n",
    //                   ost->file_index, ost->st->index);
    //            exit_program(1);
    //        }
    //        ret = av_fifo_realloc2(ost->muxing_queue, new_size);
    //        if (ret < 0)
    //            exit_program(1);
    //    }
    //    ret = av_packet_make_refcounted(pkt);
    //    if (ret < 0)
    //        exit_program(1);
    //    av_packet_move_ref(&tmp_pkt, pkt);
    //    av_fifo_generic_write(ost->muxing_queue, &tmp_pkt, sizeof(tmp_pkt), NULL);
    //    return;
    //}

    if ((st->codecpar->codec_type == AVMEDIA_TYPE_VIDEO && vsync_method_ == VSYNC_DROP))// ||
        //(st->codecpar->codec_type == AVMEDIA_TYPE_AUDIO && audio_sync_method < 0))
        pkt->pts = pkt->dts = AV_NOPTS_VALUE;

    //if (st->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
    //    int i;
    //    uint8_t *sd = av_packet_get_side_data(pkt, AV_PKT_DATA_QUALITY_STATS,
    //                                          NULL);
    //    ost->quality = sd ? AV_RL32(sd) : -1;
    //    ost->pict_type = sd ? sd[4] : AV_PICTURE_TYPE_NONE;

    //    for (i = 0; i<FF_ARRAY_ELEMS(ost->error); i++) {
    //        if (sd && i < sd[5])
    //            ost->error[i] = AV_RL64(sd + 8 + 8*i);
    //        else
    //            ost->error[i] = -1;
    //    }

    //    if (ost->frame_rate.num && ost->is_cfr) {
    //        if (pkt->duration > 0)
    //            av_log(NULL, AV_LOG_WARNING, "Overriding packet duration by frame rate, this should not happen\n");
    //        pkt->duration = av_rescale_q(1, av_inv_q(ost->frame_rate),
    //                                     ost->mux_timebase);
    //    }
    //}

    //av_packet_rescale_ts(pkt, ost->mux_timebase, ost->st->time_base);

    if (!(s->oformat->flags & AVFMT_NOTIMESTAMPS)) {
        if (pkt->dts != AV_NOPTS_VALUE &&
            pkt->pts != AV_NOPTS_VALUE &&
            pkt->dts > pkt->pts) {
            av_log(s, AV_LOG_WARNING, "Invalid DTS: %" PRId64 " PTS: %" PRId64 " in output stream %d:%d, replacing by guess\n",
                   pkt->dts, pkt->pts,
                   idx, st->index);
            pkt->pts =
            pkt->dts = pkt->pts + pkt->dts + ost->last_mux_dts + 1
                     - FFMIN3(pkt->pts, pkt->dts, ost->last_mux_dts + 1)
                     - FFMAX3(pkt->pts, pkt->dts, ost->last_mux_dts + 1);
        }
        if ((st->codecpar->codec_type == AVMEDIA_TYPE_AUDIO || st->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) &&
            pkt->dts != AV_NOPTS_VALUE &&
            !(st->codecpar->codec_id == AV_CODEC_ID_VP9 && !ost->encoding_needed) &&
            ost->last_mux_dts != AV_NOPTS_VALUE) {
            int64_t max = ost->last_mux_dts + !(s->oformat->flags & AVFMT_TS_NONSTRICT);
            if (pkt->dts < max) {
                int loglevel = max - pkt->dts > 2 || st->codecpar->codec_type == AVMEDIA_TYPE_VIDEO ? AV_LOG_WARNING : AV_LOG_DEBUG;
                av_log(s, loglevel, "Non-monotonous DTS in output stream "
                       "%d:%d; previous: %" PRId64 ", current: %" PRId64 "; ",
                       idx, st->index, ost->last_mux_dts, pkt->dts);
                //if (exit_on_error) {
                //    av_log(NULL, AV_LOG_FATAL, "aborting.\n");
                //    exit_program(1);
                //}
                av_log(s, loglevel, "changing to %" PRId64 ". This may result "
                       "in incorrect timestamps in the output file.\n",
                       max);
                if (pkt->pts >= pkt->dts)
                    pkt->pts = FFMAX(pkt->pts, max);
                pkt->dts = max;
            }
        }
    }
    ost->last_mux_dts = pkt->dts;

    ost->data_size += pkt->size;
    ost->packets_written++;

    pkt->stream_index = streams_idx_[idx];
    if (ost->encoding_needed) //not streamcopy
        av_packet_rescale_ts(pkt,
                             enc_ctxs_[idx]->time_base,
                             output_stream_[idx]->time_base);

    ret = av_interleaved_write_frame(output_fmt_ctx_, pkt);
    if (ret < 0)
        BMFLOG_NODE(BMF_ERROR, node_id_) << "Interleaved write error";
    if (!ost->encoding_needed)
        av_packet_unref(pkt);

    return ret;
}

int CFFEncoder::encode_and_write(AVFrame *frame, unsigned int idx, int *got_packet) {
    int ret;
    int got_packet_local;
    int av_index;
    OutputStream *ost = &ost_[idx];

    if (!got_packet)
        got_packet = &got_packet_local;

    av_index = (codecs_[idx]->type == AVMEDIA_TYPE_VIDEO) ? 0 : 1;
    if (av_index == 0 && frame && frame->pts < 0) {
        BMFLOG_NODE(BMF_ERROR, node_id_) << "Drop negative pts frame";
        return 0;
    }
    if (av_index == 0 && fps_ != 0 && frame) {
        frame->pts = last_pts_ + 1;
        current_frame_pts_ = frame->pts;
        ++last_pts_;
    }

    if (av_index == 0 && frame && oformat_ == "image2pipe" && push_output_) { //only support to carry orig pts time for images
        std::string stime = "";
        if (frame->metadata) {
            AVDictionaryEntry *tag = NULL;
            while ((tag = av_dict_get(frame->metadata, "", tag, AV_DICT_IGNORE_SUFFIX))) {
                if (!strcmp(tag->key, "orig_pts_time")) {
                    stime = tag->value;
                    break;
                }
            }
        }
        if (stime != "") {
            orig_pts_time_list_.push_back(std::stod(stime));
            recorded_pts_ = frame->pts;
            last_orig_pts_time_ = std::stod(stime);
        } else {
            if (recorded_pts_ >= 0)
                estimated_time_ = last_orig_pts_time_ +
                                (frame->pts - recorded_pts_) * av_q2d(enc_ctxs_[idx]->time_base);
            else
                estimated_time_ += 0.001;

            orig_pts_time_list_.push_back(estimated_time_);
        }
    }

    if(frame && enc_ctxs_[idx])
        frame->quality = enc_ctxs_[idx]->global_quality;

    ret = avcodec_send_frame(enc_ctxs_[idx], frame);
    if (ret != AVERROR_EOF && ret < 0) {
        std::string msg = "avcodec_send_frame failed: " + error_msg(ret);
        BMF_Error(BMF_TranscodeError, msg.c_str());
        return ret;
    }

    auto flush_cache = [this] () -> int {
        int ret = 0;
        while (cache_.size()) {
            auto tmp = cache_.front();
            cache_.erase(cache_.begin());
            ret = handle_output(tmp.first, tmp.second);
            av_packet_free(&tmp.first);
            if (ret < 0)
                return ret;
        }
        return ret;
    };

    while (1) {
        AVPacket *enc_pkt = av_packet_alloc();
        if (!enc_pkt) {
            BMFLOG_NODE(BMF_ERROR, node_id_) << "Pkt alloc error";
            return -1;
        }
        enc_pkt->data = NULL;
        enc_pkt->size = 0;
        av_init_packet(enc_pkt);
        *got_packet = avcodec_receive_packet(enc_ctxs_[idx], enc_pkt);
        if (*got_packet == AVERROR(EAGAIN) || *got_packet == AVERROR_EOF) {
            av_packet_free(&enc_pkt);
            if (*got_packet == AVERROR_EOF) {
                if (!stream_inited_) {
                    BMFLOG_NODE(BMF_WARNING, node_id_) << "The stream at index:" << idx << " ends, "
                        "but not all streams are initialized, all packets may be dropped.";
                    return 0;
                }
                if (ret = flush_cache(); ret < 0)
                    return ret;
            }
            ret = 0;
            break;
        }
        if ((*got_packet) != 0) {
            BMFLOG_NODE(BMF_ERROR, node_id_) << "Encode error: " << *got_packet;
            av_packet_free(&enc_pkt);
            return *got_packet;
        }

        if (push_output_ == OutputMode::OUTPUT_UNMUX_PACKET) {
            if (first_packet_[idx]) {
                auto stream = std::make_shared<AVStream>();
                *stream = *(output_stream_[idx]);
                stream->codecpar = avcodec_parameters_alloc();
                avcodec_parameters_copy(stream->codecpar, output_stream_[idx]->codecpar);
                auto packet = Packet(stream);
                //packet.set_data_type(DATA_TYPE_C);
                //packet.set_class_name("AVStream");
                if (current_task_ptr_->get_outputs().find(idx) != current_task_ptr_->get_outputs().end())
                    current_task_ptr_->get_outputs()[idx]->push(packet);
                first_packet_[idx] = false;
            }

            BMFAVPacket packet_tmp = ffmpeg::to_bmf_av_packet(enc_pkt, true);
            auto packet = Packet(packet_tmp);
            packet.set_timestamp(enc_pkt->pts * av_q2d(output_stream_[idx]->time_base) * 1000000);
            //packet.set_data_type(DATA_TYPE_C);
            //packet.set_data_class_type(BMFAVPACKET_TYPE);
            //packet.set_class_name("libbmf_module_sdk.BMFAVPacket");
            if (current_task_ptr_->get_outputs().find(idx) != current_task_ptr_->get_outputs().end())
                current_task_ptr_->get_outputs()[idx]->push(packet);
        } else {
            if (!stream_inited_ && *got_packet == 0) {
                cache_.push_back(std::pair<AVPacket*, int>(enc_pkt, idx));
                continue;
            }
            if (ret = flush_cache(); ret < 0)
                return ret;

            ret = handle_output(enc_pkt, idx);
            if (ret != 0) {
                av_packet_free(&enc_pkt);
                return ret;
            }
        }

        av_packet_free(&enc_pkt);
    }
    ost->frame_number++;
    return ret;
}

int CFFEncoder::init_stream() {
    int ret = 0;
    if (!output_fmt_ctx_)
        return 0;
    if (push_output_ == OutputMode::OUTPUT_NOTHING && !(output_fmt_ctx_->oformat->flags & AVFMT_NOFILE)) {
        ret = avio_open(&output_fmt_ctx_->pb, output_path_.c_str(), AVIO_FLAG_WRITE);
        if (ret < 0) {
            av_log(NULL, AV_LOG_ERROR, "Could not open output file '%s'", output_path_.c_str());
            return ret;
        }
    }

    if (push_output_ == OutputMode::OUTPUT_NOTHING or push_output_ == OutputMode::OUTPUT_MUXED_PACKET) {
        AVDictionary *opts = NULL;
        std::vector<std::pair<std::string, std::string>> params;
        mux_params_.get_iterated(params);
        for (int i = 0; i < params.size(); i++) {
            av_dict_set(&opts, params[i].first.c_str(), params[i].second.c_str(), 0);
        }
        {
            std::vector<std::pair<std::string, std::string>> params;
            metadata_params_.get_iterated(params);
            for (int i = 0; i < params.size(); i++) {
                av_dict_set(&output_fmt_ctx_->metadata, params[i].first.c_str(), params[i].second.c_str(), 0);
            }
        }
        ret = avformat_write_header(output_fmt_ctx_, &opts);
        if (ret < 0) {
            BMFLOG_NODE(BMF_ERROR, node_id_) << "Error occurred when opening output file";
            return ret;
        } else if (av_dict_count(opts) > 0) {
            AVDictionaryEntry *t = NULL;
            std::string err_msg = "Encoder mux_params contains incorrect key :";
            while ((t = av_dict_get(opts, "", t, AV_DICT_IGNORE_SUFFIX))) {
                err_msg.append(" ");
                err_msg.append(t->key);
            }
            av_dict_free(&opts);
            BMFLOG_NODE(BMF_ERROR, node_id_) << err_msg;
        }
        av_dict_free(&opts);

        av_dump_format(output_fmt_ctx_, 0, output_path_.c_str(), 1);
    }

    if (video_first_pts_ < audio_first_pts_) {
        first_pts_ = video_first_pts_;
    } else {
        first_pts_ = audio_first_pts_;
    }
    stream_inited_ = true;
    return 0;
}

int write_data(void *opaque, uint8_t *buf, int buf_size) {
    return ((CFFEncoder *) opaque)->write_output_data(opaque, buf, buf_size);
}

int CFFEncoder::write_current_packet_data(uint8_t *buf, int buf_size) {
    void* data = nullptr;
    AVPacket *avpkt = av_packet_alloc();
    av_init_packet(avpkt);
    av_new_packet(avpkt, buf_size);
    data = avpkt->data;
    BMFAVPacket bmf_avpkt = ffmpeg::to_bmf_av_packet(avpkt, true);

    memcpy(data, buf, buf_size);
    bmf_avpkt.set_offset(current_offset_);
    bmf_avpkt.set_whence(current_whence_);
    auto packet = Packet(bmf_avpkt);
    packet.set_timestamp(current_frame_pts_);
    packet.set_time(orig_pts_time_);
    //packet.set_data(packet_tmp);
    //packet.set_data_type(DATA_TYPE_C);
    //packet.set_data_class_type(BMFAVPACKET_TYPE);
    //packet.set_class_name("libbmf_module_sdk.BMFAVPacket");
    if (current_task_ptr_->get_outputs().find(0) != current_task_ptr_->get_outputs().end())
        current_task_ptr_->get_outputs()[0]->push(packet);

    return buf_size;
}

int CFFEncoder::write_output_data(void *opaque, uint8_t *buf, int buf_size) {
    if (oformat_ == "image2pipe" && codec_names_[0] == "mjpeg") {
        bool has_header = buf_size >= 2 && buf[0] == 0xFF && buf[1] == 0xD8;
        bool has_trailer = buf_size >= 2 && buf[buf_size - 2] == 0xFF && buf[buf_size - 1] == 0xD9;
        if (!current_image_buffer_.is_packing && has_header && has_trailer)
            return write_current_packet_data(buf, buf_size);

        if (current_image_buffer_.room - current_image_buffer_.size < buf_size) {
            current_image_buffer_.buf = (uint8_t*)av_fast_realloc(current_image_buffer_.buf,
                                                                        &current_image_buffer_.room,
                                                                        current_image_buffer_.size + buf_size);
            if (!current_image_buffer_.buf) {
                BMFLOG_NODE(BMF_ERROR, node_id_) << "Could realloc buffer for image2pipe output";
               return AVERROR(ENOMEM);
            }
        }
        memcpy(current_image_buffer_.buf + current_image_buffer_.size, buf, buf_size);
        current_image_buffer_.size += buf_size;

        if (current_image_buffer_.is_packing) {
            uint8_t *buffer = current_image_buffer_.buf;
            if (current_image_buffer_.size >= 4 &&
                buffer[0] == 0xFF && buffer[1] == 0xD8 &&
                buffer[current_image_buffer_.size - 2] == 0xFF &&
                buffer[current_image_buffer_.size - 1] == 0xD9) {

                write_current_packet_data(buffer, current_image_buffer_.size);
                current_image_buffer_.is_packing = false;
                current_image_buffer_.size = 0;
                return buf_size;
            }
        } else {
            current_image_buffer_.is_packing = true;
            return buf_size;
        }
    } else
        return write_current_packet_data(buf, buf_size);

    return buf_size;
}

int64_t seek_data(void *opaque, int64_t offset, int whence) {
    return ((CFFEncoder *) opaque)->seek_output_data(opaque, offset, whence);
}

int64_t CFFEncoder::seek_output_data(void *opaque, int64_t offset, int whence) {
    current_offset_ = offset;
    current_whence_ = whence;
    return 0;
}

int CFFEncoder::init_codec(int idx, AVFrame* frame) {
    AVStream *out_stream;
    int ret;

    codec_init_touched_num_++;

    if (!frame && ost_[idx].encoding_needed) {
        ret = 0;
        if (codec_init_touched_num_ == num_input_streams_)
            ret = init_stream();
        return ret;
    }

    if (!output_fmt_ctx_) {
        avformat_alloc_output_context2(&output_fmt_ctx_, NULL, (oformat_ != "" ? oformat_.c_str() : NULL),
                                           push_output_ == OutputMode::OUTPUT_NOTHING ? output_path_.c_str() : NULL);
        if (!output_fmt_ctx_) {
            BMFLOG_NODE(BMF_ERROR, node_id_) << "Could not create output context";
            return AVERROR_UNKNOWN;
        }
        if (push_output_ == OutputMode::OUTPUT_MUXED_PACKET) {
            unsigned char *avio_ctx_buffer;
            size_t avio_ctx_buffer_size = avio_buffer_size_;
            avio_ctx_buffer = (unsigned char*)av_malloc(avio_ctx_buffer_size);
            if (!avio_ctx_buffer) {
                BMFLOG_NODE(BMF_ERROR, node_id_) << "Could not create avio buffer";
                return AVERROR_UNKNOWN;
            }
            avio_ctx_ = avio_alloc_context(avio_ctx_buffer, avio_ctx_buffer_size, 1, (void*) this, NULL, write_data, seek_data);
            avio_ctx_->seekable = AVIO_SEEKABLE_NORMAL;
            output_fmt_ctx_->pb = avio_ctx_;
            output_fmt_ctx_->flags = AVFMT_FLAG_CUSTOM_IO;
            current_image_buffer_.buf = (unsigned char*)av_malloc(avio_ctx_buffer_size);
            current_image_buffer_.room = avio_ctx_buffer_size;
            current_image_buffer_.size = 0;
        }
    }

    if (ost_[idx].encoding_needed) {
        codecs_[idx] = avcodec_find_encoder_by_name(codec_names_[idx].c_str());
        if (!codecs_[idx]) {
            BMFLOG_NODE(BMF_ERROR, node_id_) << "Codec '" << codec_names_[idx].c_str() << "' not found";
            return AVERROR_UNKNOWN;
        }
    }
    enc_ctxs_[idx] = avcodec_alloc_context3(codecs_[idx]);
    if (!enc_ctxs_[idx]) {
        BMFLOG_NODE(BMF_ERROR, node_id_) << "Failed to allocate the encoder context";
        return AVERROR(ENOMEM);
    }

    output_stream_[idx] = avformat_new_stream(output_fmt_ctx_, NULL);
    if (!output_stream_[idx]) {
        BMFLOG_NODE(BMF_ERROR, node_id_) << "Could not allocate stream";
        return AVERROR_UNKNOWN;
    }

    out_stream = output_stream_[idx];
    streams_idx_[idx] = out_stream->index;

    if (!ost_[idx].encoding_needed) {
        AVCodecParameters *par_dst = out_stream->codecpar;
        AVCodecParameters *par_src = avcodec_parameters_alloc();
        uint32_t codec_tag = par_dst->codec_tag;
        auto input_stream = ost_[idx].input_stream;

        if (!input_stream) {
            BMFLOG_NODE(BMF_ERROR, node_id_) << "input stream info is needed for stream copy";
            return AVERROR_INVALIDDATA;
        }

        ret = avcodec_parameters_to_context(enc_ctxs_[idx], input_stream->codecpar);
        avcodec_parameters_free(&input_stream->codecpar);
        //if (ret >= 0)
        //    ret = av_opt_set_dict(enc_ctxs_[idx], &ost_[idx].encoder_opts);
        if (ret < 0) {
            av_log(NULL, AV_LOG_FATAL,
                   "Error setting up codec context options.\n");
            avcodec_parameters_free(&par_src);
            return ret;
        }
        enc_ctxs_[idx]->time_base = input_stream->time_base;

        ret = avcodec_parameters_from_context(par_src, enc_ctxs_[idx]);
        if (ret < 0) {
            avcodec_parameters_free(&par_src);
            std::string msg = "avcodec_parameters_from_context failed: " + error_msg(ret);
            BMF_Error(BMF_TranscodeError, msg.c_str());
        }

        if (!codec_tag) {
            unsigned int codec_tag_tmp;
            if (!output_fmt_ctx_->oformat->codec_tag ||
                av_codec_get_id(output_fmt_ctx_->oformat->codec_tag, par_src->codec_tag) == par_src->codec_id ||
                !av_codec_get_tag2(output_fmt_ctx_->oformat->codec_tag, par_src->codec_id, &codec_tag_tmp))
                codec_tag = par_src->codec_tag;
        }

        ret = avcodec_parameters_copy(par_dst, par_src);
        if (ret < 0) {
            avcodec_parameters_free(&par_src);
            return ret;
        }
        par_dst->codec_tag = codec_tag;

        AVRational sar;
        switch (par_dst->codec_type) {
        case AVMEDIA_TYPE_AUDIO:
            //if (audio_volume != 256) {
            //    av_log(NULL, AV_LOG_FATAL, "-acodec copy and -vol are incompatible (frames are not decoded)\n");
            //    exit_program(1);
            //}
            if((par_dst->block_align == 1 || par_dst->block_align == 1152 || par_dst->block_align == 576) && par_dst->codec_id == AV_CODEC_ID_MP3)
                par_dst->block_align= 0;
            if(par_dst->codec_id == AV_CODEC_ID_AC3)
                par_dst->block_align= 0;
            break;
        case AVMEDIA_TYPE_VIDEO:
            if (input_stream->sample_aspect_ratio.num)
                sar = input_stream->sample_aspect_ratio;
            else
                sar = par_src->sample_aspect_ratio;
            out_stream->sample_aspect_ratio = par_dst->sample_aspect_ratio = sar;
            out_stream->avg_frame_rate = input_stream->avg_frame_rate;
            out_stream->r_frame_rate = input_stream->r_frame_rate;
            break;
        }

        out_stream->time_base = input_stream->time_base;
        out_stream->codec->time_base = input_stream->time_base;
        ret = 0;
        if (num_input_streams_ == codec_init_touched_num_)
            ret = init_stream();
        avcodec_parameters_free(&par_src);
        return ret;
    }

    AVDictionary *enc_opts = NULL;
    if (idx == 0) {
        if (frame and frame->pts != AV_NOPTS_VALUE) {
            video_first_pts_ = frame->pts;
        }
        enc_ctxs_[idx]->pix_fmt = pix_fmt_;
        if (width_ && height_) {
            enc_ctxs_[idx]->width = width_;
            enc_ctxs_[idx]->height = height_;
        } else if (b_stream_eof_[idx] || !frame) {
            enc_ctxs_[idx]->width = 320;
            enc_ctxs_[idx]->height = 240;
        } else {
            enc_ctxs_[idx]->width = frame->width;
            enc_ctxs_[idx]->height = frame->height;
        }
        enc_ctxs_[idx]->bit_rate = 0;
        enc_ctxs_[idx]->time_base = (AVRational){1, 1000000};
        if (frame && frame->metadata) {
            AVDictionaryEntry *tag = NULL;
            while ((tag = av_dict_get(frame->metadata, "", tag, AV_DICT_IGNORE_SUFFIX))) {
                if (!strcmp(tag->key, "time_base")) {
                    std::string svalue = tag->value;
                    int pos = svalue.find(",");
                    if (pos > 0) {
                        AVRational r;
                        r.num = stoi(svalue.substr(0, pos));
                        r.den = stoi(svalue.substr(pos + 1));
                        in_stream_tbs_[idx] = r;
                    }
                }
                if (!strcmp(tag->key, "frame_rate")) {
                    std::string svalue = tag->value;
                    int pos = svalue.find(",");
                    if (pos > 0) {
                        AVRational r;
                        r.num = stoi(svalue.substr(0, pos));
                        r.den = stoi(svalue.substr(pos + 1));
                        input_video_frame_rate_ = r;
                        if (input_video_frame_rate_.num > 0 && input_video_frame_rate_.den > 0)
                            video_frame_rate_ = input_video_frame_rate_;
                    }
                }
                if (!strcmp(tag->key, "sample_aspect_ratio")) {
                    std::string svalue = tag->value;
                    int pos = svalue.find(",");
                    if (pos > 0) {
                        AVRational r;
                        r.num = stoi(svalue.substr(0, pos));
                        r.den = stoi(svalue.substr(pos + 1));
                        input_sample_aspect_ratio_ = r;
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

                if (!strcmp(tag->key, "copyts"))
                    copy_ts_ = true;
            }
        }
        /** @addtogroup EncM
         * @{
         * @arg threads: specify the number of threads for encoder, "auto" by default
         * @} */
        if (!video_params_.has_key("threads")) {
            av_dict_set(&enc_opts, "threads", "auto", 0);
        }

        /** @addtogroup EncM
         * @{
         * @arg psnr: to set encoder provide psnr information
         * @} */
        if (video_params_.has_key("psnr")) {
            int do_psnr;
            video_params_.get_int("psnr", do_psnr);
            video_params_.erase("psnr");
            if (do_psnr != 0)
                enc_ctxs_[0]->flags |= AV_CODEC_FLAG_PSNR;
        }
        
        /** @addtogroup EncM
         * @{
         * @arg in_time_base: to set time base manually
         * @} */
        if (video_params_.has_key("in_time_base")) {
            std::string svalue;
            video_params_.get_string("in_time_base",svalue);
            video_params_.erase("in_time_base");
            int pos = svalue.find(",");
            if (pos > 0) {
                AVRational r;
                r.num = stoi(svalue.substr(0, pos));
                r.den = stoi(svalue.substr(pos + 1));
                in_stream_tbs_[idx] = r;
            }
        }

        /** @addtogroup EncM
         * @{
         * @arg vsync: to set the video sync method on frame rate, "auto" by default.
         * and it can be "cfr", "vfr", "passthrough", "drop" similar as ffmpeg
         * @} */
        vsync_method_ = VSYNC_AUTO;
        if (video_params_.has_key("vsync")) {
            std::string s_vsync;
            video_params_.get_string("vsync", s_vsync);
            video_params_.erase("vsync");
            if (!av_strcasecmp(s_vsync.c_str(), "cfr"))
                vsync_method_ = VSYNC_CFR;
            else if (!av_strcasecmp(s_vsync.c_str(), "vfr"))
                vsync_method_ = VSYNC_VFR;
            else if (!av_strcasecmp(s_vsync.c_str(), "passthrough"))
                vsync_method_ = VSYNC_PASSTHROUGH;
            else if (!av_strcasecmp(s_vsync.c_str(), "drop"))
                vsync_method_ = VSYNC_DROP;
        }
        /** @addtogroup EncM
         * @{
         * @arg max_fr: to set the frame rate
         * @} */
        if (video_params_.has_key("max_fr")) {
            double max_fr;
            std::string max_fr_string;
            video_params_.get_double("max_fr", max_fr);
            max_fr_string = std::to_string(max_fr);
            AVRational frame_rate;
            av_parse_video_rate(&frame_rate, max_fr_string.c_str());
            //the frame rate set by encode user
            video_frame_rate_ = frame_rate;
            video_params_.erase("max_fr");

            if (vsync_method_ == VSYNC_AUTO)
                vsync_method_ = VSYNC_VFR;
        }
        /** @addtogroup EncM
         * @{
         * @arg max_fr: to set the frame rate, similar as ffmpeg
         * @} */
        if (video_params_.has_key("r")) {
            double fr;
            std::string fr_string;
            if (video_params_.json_value_["r"].is_string()) {
                video_params_.get_string("r",fr_string);
            }
            else {
                video_params_.get_double("r", fr);
                fr_string = std::to_string(fr);
            }
            AVRational frame_rate;
            av_parse_video_rate(&frame_rate, fr_string.c_str());
            //the frame rate set by encode user
            video_frame_rate_ = frame_rate;
            video_params_.erase("r");
        }

        if (vsync_method_ == VSYNC_AUTO) {
            if(!strcmp(output_fmt_ctx_->oformat->name, "avi"))
                vsync_method_ = VSYNC_VFR;
            else
                vsync_method_ = (output_fmt_ctx_->oformat->flags & AVFMT_VARIABLE_FPS) ?
                                 ((output_fmt_ctx_->oformat->flags & AVFMT_NOTIMESTAMPS) ?
                                   VSYNC_PASSTHROUGH : VSYNC_VFR) : VSYNC_CFR;
            //if (   ist
            //    && vsync_method_ == VSYNC_CFR
            //    && input_files[ist->file_index]->ctx->nb_streams == 1
            //    && input_files[ist->file_index]->input_ts_offset == 0) {
            //    vsync_method_ = VSYNC_VSCFR;
            //}
            if (vsync_method_ == VSYNC_CFR && copy_ts_)
                vsync_method_ = VSYNC_VSCFR;
        }

        if (video_frame_rate_.num == 0) {
            video_frame_rate_.num = 25;
            video_frame_rate_.den = 1;
        }
        enc_ctxs_[idx]->time_base.num = video_frame_rate_.den;
        enc_ctxs_[idx]->time_base.den = video_frame_rate_.num;
        if (in_stream_tbs_[idx].num <= 0 && in_stream_tbs_[idx].den <= 0)
            in_stream_tbs_[idx] = enc_ctxs_[idx]->time_base;

        if (input_sample_aspect_ratio_.num > 0 && input_sample_aspect_ratio_.den > 0) {
            enc_ctxs_[idx]->sample_aspect_ratio = input_sample_aspect_ratio_;
            output_stream_[idx]->sample_aspect_ratio = enc_ctxs_[idx]->sample_aspect_ratio;
        } else {
            enc_ctxs_[idx]->sample_aspect_ratio = frame->sample_aspect_ratio;
            output_stream_[idx]->sample_aspect_ratio = enc_ctxs_[idx]->sample_aspect_ratio;
        }

        /** @addtogroup EncM
         * @{
         * @arg qscal: to set the qscale for the encoder global_quality
         * @} */
        double qscale = -1;
        if (video_params_.has_key("qscale")) {
            video_params_.get_double("qscale", qscale);
            video_params_.erase("qscale");
        }
        if (qscale > 0) {
            enc_ctxs_[idx]->flags |= AV_CODEC_FLAG_QSCALE;
            enc_ctxs_[idx]->global_quality = FF_QP2LAMBDA * qscale;
        }

        /** @addtogroup EncM
         * @{
         * @arg vtag: to set the vtag for output stream
         * @} */
        if (video_params_.has_key("vtag")) {
            video_params_.get_string("vtag", vtag_);
            video_params_.erase("vtag");
        }

        /** @addtogroup EncM
         * @{
         * @arg bit_rate or b: to set the bitrate for video encode
         * @} */
        if (video_params_.has_key("bit_rate")) {
            video_params_.get_long("bit_rate", enc_ctxs_[idx]->bit_rate);
            video_params_.erase("bit_rate");
        }

        std::vector<std::pair<std::string, std::string>> params;
        video_params_.get_iterated(params);
        for (int i = 0; i < params.size(); i++) {
            av_dict_set(&enc_opts, params[i].first.c_str(), params[i].second.c_str(), 0);
        }
    }
    if (idx == 1) {
        if (frame and frame->pts != AV_NOPTS_VALUE) {
            audio_first_pts_ = frame->pts;
        }
        enc_ctxs_[idx]->bit_rate = 0;
        enc_ctxs_[idx]->sample_rate = 44100;
        enc_ctxs_[idx]->time_base = (AVRational){1, 44100};
        enc_ctxs_[idx]->channels = 2;
        enc_ctxs_[idx]->channel_layout = av_get_default_channel_layout(enc_ctxs_[idx]->channels);

        /** @addtogroup EncM
         * @{
         * @arg channels: to set the channels for input audio
         * @} */
        if (audio_params_.has_key("channels")) {
            audio_params_.get_int("channels", enc_ctxs_[idx]->channels);
            audio_params_.erase("channels");
            enc_ctxs_[idx]->channel_layout = av_get_default_channel_layout(enc_ctxs_[idx]->channels);
        } else {
            if (frame && frame->channels) {
                enc_ctxs_[idx]->channels = frame->channels;
                if (frame->channel_layout)
                    enc_ctxs_[idx]->channel_layout = frame->channel_layout;
                else {
                    enc_ctxs_[idx]->channel_layout = av_get_default_channel_layout(frame->channels);
                    frame->channel_layout = enc_ctxs_[idx]->channel_layout;
                }
            } else if (frame && frame->channel_layout) {
                enc_ctxs_[idx]->channel_layout = frame->channel_layout;
                enc_ctxs_[idx]->channels = av_get_channel_layout_nb_channels(enc_ctxs_[idx]->channel_layout);
            }
        }

        /** @addtogroup EncM
         * @{
         * @arg bit_rate or b: to set the bit_rate for audio encode
         * @} */
        if (audio_params_.has_key("bit_rate")) {
            audio_params_.get_long("bit_rate", enc_ctxs_[idx]->bit_rate);
            audio_params_.erase("bit_rate");
        }

        /** @addtogroup EncM
         * @{
         * @arg sample_rate: to set the sample_rate for audio encode
         * @} */
        //set the time base of audio encoder to be same as 1 / encoder sample_rate
        if (audio_params_.has_key("sample_rate")) {
            audio_params_.get_int("sample_rate", enc_ctxs_[idx]->sample_rate);
            enc_ctxs_[idx]->time_base = (AVRational){1, enc_ctxs_[idx]->sample_rate};
            audio_params_.erase("sample_rate");
        } else if (frame && frame->sample_rate) {
            enc_ctxs_[idx]->sample_rate = frame->sample_rate;
            enc_ctxs_[idx]->time_base = (AVRational){1, frame->sample_rate};
        }

        if (frame && frame->sample_rate) {
            in_stream_tbs_[idx] = (AVRational){1, frame->sample_rate};
        }
        if (frame && frame->metadata) {
            AVDictionaryEntry *tag = NULL;
            while ((tag = av_dict_get(frame->metadata, "", tag, AV_DICT_IGNORE_SUFFIX)))
                if (!strcmp(tag->key, "time_base")) {
                    std::string svalue = tag->value;
                    int pos = svalue.find(",");
                    if (pos > 0) {
                        AVRational r;
                        r.num = stoi(svalue.substr(0, pos));
                        r.den = stoi(svalue.substr(pos + 1));
                        in_stream_tbs_[idx] = r;
                    }
                }
        }

        double qscale = -1;
        if (audio_params_.has_key("qscale")) {
            audio_params_.get_double("qscale", qscale);
            audio_params_.erase("qscale");
        }
        if (qscale > 0) {
            enc_ctxs_[idx]->flags |= AV_CODEC_FLAG_QSCALE;
            enc_ctxs_[idx]->global_quality = FF_QP2LAMBDA * qscale;
        }

        /** @addtogroup EncM
         * @{
         * @arg atag: to set the atag for output stream
         * @} */
        if (audio_params_.has_key("atag")) {
            audio_params_.get_string("atag", atag_);
            audio_params_.erase("atag");
        }
        enc_ctxs_[idx]->sample_fmt = codecs_[idx]->sample_fmts[0];

        std::vector<std::pair<std::string, std::string>> params;
        audio_params_.get_iterated(params);
        for (int i = 0; i < params.size(); i++) {
            av_dict_set(&enc_opts, params[i].first.c_str(), params[i].second.c_str(), 0);
        }
    }

    if (output_fmt_ctx_->oformat->flags & AVFMT_GLOBALHEADER)
        enc_ctxs_[idx]->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
    if (frame && frame->hw_frames_ctx) {
        AVHWFramesContext* hw_frames_context = (AVHWFramesContext*)frame->hw_frames_ctx->data;
        enc_ctxs_[idx]->hw_device_ctx = av_buffer_ref(hw_frames_context->device_ref);
        enc_ctxs_[idx]->hw_frames_ctx = av_buffer_ref(frame->hw_frames_ctx);
    }
    ret = avcodec_open2(enc_ctxs_[idx], codecs_[idx], &enc_opts);
    if (ret < 0) {
        BMFLOG_NODE(BMF_ERROR, node_id_) << "avcodec_open2 result: " << ret;
        BMFLOG_NODE(BMF_ERROR, node_id_) << "Cannot open video/audio encoder for stream #" << idx;
        return ret;
    } else if (av_dict_count(enc_opts) > 0) {
        AVDictionaryEntry *t = NULL;
        std::string err_msg;
        if (idx == 0)
            err_msg = "Encoder video_params contains incorrect key :";
        else if (idx == 1)
            err_msg = "Encoder audio_params contains incorrect key :";
        while ((t = av_dict_get(enc_opts, "", t, AV_DICT_IGNORE_SUFFIX))) {
            err_msg.append(" ");
            err_msg.append(t->key);
        }
        av_dict_free(&enc_opts);
        BMFLOG_NODE(BMF_ERROR, node_id_) << err_msg;
    }
    av_dict_free(&enc_opts);

    ret = avcodec_parameters_from_context(out_stream->codecpar, enc_ctxs_[idx]);
    if (ret < 0) {
        BMFLOG_NODE(BMF_ERROR, node_id_) << "Failed to copy encoder parameters to output stream";
        return ret;
    }

    if (idx == 0) {
        if (not vtag_.empty()) {
            char *next;
            uint32_t tag = strtol(vtag_.c_str(), &next, 0);
            if (*next)
                tag = AV_RL32(vtag_.c_str());
            out_stream->codecpar->codec_tag =
            enc_ctxs_[idx]->codec_tag = tag;
        }

        enc_ctxs_[0]->framerate = video_frame_rate_;
        out_stream->r_frame_rate = video_frame_rate_;
        out_stream->avg_frame_rate = video_frame_rate_;
    }
    if (idx == 1) {
        if (not atag_.empty()) {
            char *next;
            uint32_t tag = strtol(atag_.c_str(), &next, 0);
            if (*next)
                tag = AV_RL32(atag_.c_str());
            out_stream->codecpar->codec_tag =
            enc_ctxs_[idx]->codec_tag = tag;
        }
    }

    out_stream->time_base = enc_ctxs_[idx]->time_base;
    out_stream->codec->time_base = enc_ctxs_[idx]->time_base;

    ret = 0;
    if (num_input_streams_ == codec_init_touched_num_)
        ret = init_stream();

    return ret;
}

int CFFEncoder::flush() {
    int got_packet = 0;
    int ret = 0;

    if (b_flushed_)
        return 0;

    for (int idx = 0; idx < num_input_streams_; idx++) {
        if (!codecs_[idx])
            continue;

        if (idx == 1) {
            ret = handle_audio_frame(NULL, true, 1);
            if (ret < 0)
                return ret;
        }

        while (1) {
            if (codecs_[idx]->type == AVMEDIA_TYPE_VIDEO && video_sync_) {
                std::vector<AVFrame*> sync_frames;
                video_sync_->process_video_frame(NULL, sync_frames, ost_[idx].frame_number);
                for (int j = 0; j < sync_frames.size(); j++) {
                    int got_frame = 0;
                    int ret = encode_and_write(sync_frames[j], idx, &got_frame);
                    av_frame_free(&sync_frames[j]);
                }
            }

            ret = encode_and_write(NULL, idx, &got_packet);
            if (got_packet == AVERROR(EAGAIN))
                continue;
            if (ret != AVERROR_EOF && ret < 0) {
                BMFLOG_NODE(BMF_ERROR, node_id_) << "encode and write failed ret:" << ret;
                return ret;
            }
            if (ret == AVERROR_EOF || got_packet != 0)
                break;
        }
    }

    b_flushed_ = true;
    if (output_fmt_ctx_ && (push_output_ == OutputMode::OUTPUT_NOTHING or push_output_ == OutputMode::OUTPUT_MUXED_PACKET))
        ret = av_write_trailer(output_fmt_ctx_);

    return ret;
}

int CFFEncoder::close() {
    int ret;
    flush();
    clean();
    return ret;
}

int CFFEncoder::handle_audio_frame(AVFrame *frame, bool is_flushing, int index) {
    int ret = 0;
    // if the frame is NULL and the audio_resampler is not inited, it should just return 0.
    // this situation will happen when the encoder has audio stream but receive no audio packet. 
    if (output_audio_filter_graph_ == NULL && !frame) {
        return ret;
    }
    if (output_audio_filter_graph_ == NULL && frame) {
        output_audio_filter_graph_ = std::make_shared<FilterGraph>();
        FilterConfig in_config;
        FilterConfig out_config;
        in_config.sample_rate = frame->sample_rate;
        in_config.format = frame->format;
        in_config.channels = frame->channels;
        in_config.channel_layout = frame->channel_layout;
        in_config.tb = in_stream_tbs_[index];

        out_config.sample_rate = enc_ctxs_[index]->sample_rate;
        out_config.format = enc_ctxs_[index]->sample_fmt;
        out_config.channels = enc_ctxs_[index]->channels;
        out_config.channel_layout = enc_ctxs_[index]->channel_layout;

        std::map<int, FilterConfig> in_cfgs;
        std::map<int, FilterConfig> out_cfgs;
        std::string descr = "[i0_0]anull[o0_0]";
        if (!(codecs_[index]->capabilities & AV_CODEC_CAP_VARIABLE_FRAME_SIZE))
            out_config.frame_size = enc_ctxs_[index]->frame_size;
        in_cfgs[0] = in_config;
        out_cfgs[0] = out_config;
        if (output_audio_filter_graph_->config_graph(descr, in_cfgs, out_cfgs) != 0) {
            BMFLOG_NODE(BMF_ERROR, node_id_) << "output audio filter graph config failed";
            return -1;
        }
    }

    std::vector<AVFrame*> filter_frame_list;
    ret = output_audio_filter_graph_->get_filter_frame(frame, 0, 0, filter_frame_list);
    if (ret != 0 && ret != AVERROR_EOF) {
        std::string err_msg = "Failed to inject frame into filter network, in encoder";
        BMF_Error(BMF_TranscodeFatalError, err_msg.c_str());
    }
    if (frame)
        av_frame_free(&frame);

    int got_frame = 0;
    for (int i=0; i<filter_frame_list.size(); i++) {
        AVRational tb = av_buffersink_get_time_base(output_audio_filter_graph_->buffer_sink_ctx_[0]);
        filter_frame_list[i]->pts = av_rescale_q(filter_frame_list[i]->pts, tb, enc_ctxs_[1]->time_base);
        int ret = encode_and_write(filter_frame_list[i], index, &got_frame);
        av_frame_free(&filter_frame_list[i]);
    }

    return 0;
}

bool CFFEncoder::need_output_video_filter_graph(AVFrame *frame) {
    if (width_ == 0 && height_ == 0 && frame) {
        width_ = frame->width;
        height_ = frame->height;
    }
    if (width_!=0 && height_ != 0 && (width_ != frame->width || height_ != frame->height))
        return true;
    if (frame->format != pix_fmt_)
        return true;
    return false;
}

int CFFEncoder::handle_video_frame(AVFrame *frame, bool is_flushing, int index) {
    AVFrame *resize_frame = NULL;
    int ret;
    std::vector<AVFrame*> filter_frames;
    std::vector<AVFrame*> sync_frames;

    if (output_video_filter_graph_ == NULL && need_output_video_filter_graph(frame)) {
        std::map<int, FilterConfig> in_cfgs;
        std::map<int, FilterConfig> out_cfgs;
        output_video_filter_graph_ = std::make_shared<FilterGraph>();
        FilterConfig in_config;
        FilterConfig out_config;
        in_config.width = frame->width;
        in_config.height = frame->height;
        in_config.format = frame->format;
        in_config.tb = in_stream_tbs_[index];
        in_config.sample_aspect_ratio = frame->sample_aspect_ratio;
        in_cfgs[0] = in_config;
        out_cfgs[0] = out_config;
        char args[100];
        snprintf(args, 100, "scale=%d:%d,format=pix_fmts=%s",
                 width_, height_, av_get_pix_fmt_name(pix_fmt_));
        std::string args_str = args;
        std::string descr = "[i0_0]" + args_str + "[o0_0]";
        if (output_video_filter_graph_->config_graph(descr, in_cfgs, out_cfgs) != 0) {
            BMFLOG_NODE(BMF_ERROR, node_id_) << "output video filter graph config failed";
            return -1;
        }
    }

    if (output_video_filter_graph_) {
        ret = output_video_filter_graph_->get_filter_frame(frame, 0, 0, filter_frames);
        if (ret != 0 && ret != AVERROR_EOF) {
            std::string err_msg = "Failed to inject frame into filter network, in encoder";
            BMF_Error(BMF_TranscodeFatalError, err_msg.c_str());
        }
        av_frame_free(&frame);
    }
    else
        filter_frames.push_back(frame);

    for (int i=0; i < filter_frames.size(); i++) {
        AVFrame* filter_frame = filter_frames[i];
        if (video_sync_ == NULL) {
            video_sync_ = std::make_shared<VideoSync>(in_stream_tbs_[0], enc_ctxs_[0]->time_base, input_video_frame_rate_, video_frame_rate_, stream_start_time_, stream_first_dts_, vsync_method_, ost_[0].max_frames, ost_[0].min_frames);
        }

        video_sync_->process_video_frame(filter_frame, sync_frames, ost_[0].frame_number);
        for (int j = 0; j < sync_frames.size(); j++) {
            int got_frame = 0;
            int ret = encode_and_write(sync_frames[j], index, &got_frame);
            av_frame_free(&sync_frames[j]);
        }
        sync_frames.clear();
        av_frame_free(&filter_frame);
    }
    return 0;
}

int CFFEncoder::handle_frame(AVFrame *frame, int index) {
    int got_packet;
    int ret = 0;
    frame->pict_type = AV_PICTURE_TYPE_NONE;

    if (index == 0 ) {
        handle_video_frame(frame, false, 0);
        return ret;
    } else if (index == 1) {
        if (frame->channel_layout == 0) {
            if (frame->channels)
                frame->channel_layout = av_get_default_channel_layout(frame->channels);
            else
                frame->channel_layout = enc_ctxs_[index]->channel_layout;
        }
        if (frame->channels == 0) {
            if (frame->channel_layout)
                frame->channels = av_get_channel_layout_nb_channels(frame->channel_layout);
            else
                frame->channels = enc_ctxs_[index]->channels;
        }
        ret = handle_audio_frame(frame, false, 1);
        return ret;

    }
    return 0;
}

int CFFEncoder::streamcopy(AVPacket *ipkt, AVPacket *opkt, int idx) {
    if (!ipkt || !opkt)
        return -1;
    av_init_packet(opkt);

    if (ipkt->pts != AV_NOPTS_VALUE)
        opkt->pts = av_rescale_q(ipkt->pts, enc_ctxs_[idx]->time_base, output_stream_[idx]->time_base);
    else
        opkt->pts = AV_NOPTS_VALUE;

    if (ipkt->dts != AV_NOPTS_VALUE)
        opkt->dts = av_rescale_q(ipkt->dts, enc_ctxs_[idx]->time_base, output_stream_[idx]->time_base);
    else
        opkt->dts = AV_NOPTS_VALUE;

    if (output_stream_[idx]->codecpar->codec_type == AVMEDIA_TYPE_AUDIO && ipkt->dts != AV_NOPTS_VALUE) {
        AVCodecParameters *cpar = output_stream_[idx]->codecpar;
        int duration = av_get_audio_frame_duration2(cpar, ipkt->size);
        if(!duration)
            duration = cpar->frame_size;
        opkt->dts = opkt->pts = av_rescale_delta(enc_ctxs_[idx]->time_base, ipkt->dts,
                                                 (AVRational){1, cpar->sample_rate}, duration,
                                                 &ost_[idx].filter_in_rescale_delta_last,
                                                 output_stream_[idx]->time_base);
    }

    opkt->duration = av_rescale_q(ipkt->duration, enc_ctxs_[idx]->time_base, output_stream_[idx]->time_base);

    opkt->flags    = ipkt->flags;

    if (ipkt->buf) {
        opkt->buf = av_buffer_ref(ipkt->buf);
        if (!opkt->buf)
            return -1;
    }
    opkt->data = ipkt->data;
    opkt->size = ipkt->size;

    av_copy_packet_side_data(opkt, ipkt);
    return 0;
}

int CFFEncoder::process(Task &task) {
    current_task_ptr_ = &task;
    if (reset_flag_) {
        if (check_valid_task(task)) {
            b_init_ = false;
            init();
        } else
            return PROCESS_OK;
    }

    int got_packet;
    Packet packet;
    AVFrame *frame;
    int ret = 0;

    if (!num_input_streams_)
        num_input_streams_ = task.get_inputs().size();
    while (((task.get_inputs().find(0) != task.get_inputs().end() && !task.get_inputs()[0]->empty()) ||
            (task.get_inputs().find(1) != task.get_inputs().end() && !task.get_inputs()[1]->empty())) && !b_eof_) {
        for (int index = 0; index < num_input_streams_; index++) {
            if (task.get_inputs().find(index) == task.get_inputs().end())
                continue;
            if (b_stream_eof_[index])
                continue;
            if (task.get_inputs()[index]->empty())
                continue;

            task.pop_packet_from_input_queue(index, packet);
            if (packet.timestamp() == BMF_EOF) {
                b_stream_eof_[index] = true;
                b_eof_ = num_input_streams_ == 1 ? b_stream_eof_[0] : (b_stream_eof_[0] & b_stream_eof_[1]);
                if (!enc_ctxs_[index] && !null_output_) {//at the begining the stream got EOF
                    ret = init_codec(index, NULL);
                    if (ret < 0) {
                        BMFLOG_NODE(BMF_ERROR, node_id_) << "init codec error when eof got at begining";
                        return ret;
                    }
                }
                continue;
            }

            if(packet.is<std::shared_ptr<AVStream>>()){
                ost_[index].input_stream = packet.get<std::shared_ptr<AVStream>>();
                ost_[index].encoding_needed = false;
                continue;
            }

            if (ost_[index].frame_number >= ost_[index].max_frames)
                continue;
            
            if(packet.is<BMFAVPacket>()){
                BMFAVPacket av_packet = packet.get<BMFAVPacket>();
                auto in_pkt = ffmpeg::from_bmf_av_packet(av_packet, false);
                ost_[index].encoding_needed = false;
                if (!enc_ctxs_[index]) {
                    ret = init_codec(index, NULL);
                    if (ret < 0) {
                        BMFLOG_NODE(BMF_ERROR, node_id_) << "init codec error when avpacket input";
                        return ret;
                    }
                }
                if (!stream_inited_) {
                    AVPacket *pkt = av_packet_clone(in_pkt);
                    stream_copy_cache_.push_back(std::pair<AVPacket*, int>(pkt, index));
                }
                else {
                    while (stream_copy_cache_.size()) {
                        auto tmp = stream_copy_cache_.front();
                        stream_copy_cache_.erase(stream_copy_cache_.begin());
                        ret = handle_output(tmp.first, tmp.second);
                        av_packet_free(&tmp.first);
                        if (ret < 0)
                            return ret;
                    }
                    ret = handle_output(in_pkt, index);
                }
                continue;
            }

            if (index == 0) {
                auto video_frame = packet.get<VideoFrame>();
                frame = av_frame_clone(ffmpeg::from_video_frame(video_frame, true));

                if (oformat_ == "image2pipe" && push_output_) { //only support to carry orig pts time for images
                    std::string stime = "";
                    if (frame->metadata) {
                        AVDictionaryEntry *tag = NULL;
                        while ((tag = av_dict_get(frame->metadata, "", tag, AV_DICT_IGNORE_SUFFIX))) {
                            if (!strcmp(tag->key, "orig_pts_time")) {
                                stime = tag->value;
                                break;
                            }
                        }
                    }
                    if (stime != "")
                        orig_pts_time_list_.push_back(std::stod(stime));
                }
            }
            if (index == 1) {
                auto audio_frame = packet.get<AudioFrame>();
                frame = ffmpeg::from_audio_frame(audio_frame, false);
            }

            if (null_output_) {
                if (frame)
                    av_frame_free(&frame);
                continue;
            }

            ret = av_frame_make_writable(frame);
            if (ret < 0) {
                BMFLOG_NODE(BMF_ERROR, node_id_) << "frame writable error";
                continue;
            }

            if (!codecs_[index]) {
                ret = init_codec(index, frame);
                if (ret < 0) {
                    av_frame_free(&frame);
                    BMFLOG_NODE(BMF_ERROR, node_id_) << "init codec error";
                    return ret;
                }
            }

            if (adjust_pts_flag_) {
                if (stream_inited_) {
                    while (frame_cache_.size()) {
                        auto temp = frame_cache_.front();
                        temp.first->pts = temp.first->pts - first_pts_;

                        frame_cache_.erase(frame_cache_.begin());
                        handle_frame(temp.first, temp.second);
                    }
                    frame->pts = frame->pts - first_pts_;
                    current_frame_pts_ = frame->pts;
                    handle_frame(frame, index);
                } else {
                    frame_cache_.push_back(std::pair<AVFrame *, int>(frame, index));
                }
            } else {
                handle_frame(frame, index);
            }
        }
    }

    if (b_eof_) {
        if (!null_output_) {
            if (task.get_outputs().size() > 0 && !b_flushed_) {
                if (push_output_ == OutputMode::OUTPUT_NOTHING) {
                    std::string data;
                    if (!output_dir_.empty())
                        data = output_dir_;
                    else
                        data = output_path_;
                    auto packet = Packet(data);
                    packet.set_timestamp(1);
                    task.get_outputs()[0]->push(packet);
                }
            }
            flush();
            if (task.get_outputs().size() > 0 || push_output_ != OutputMode::OUTPUT_NOTHING) {
                Packet pkt = Packet::generate_eof_packet();
                assert(pkt.timestamp_ == BMF_EOF);
                for (int i = 0; i < task.get_outputs().size(); i++){
                    task.get_outputs()[i]->push(pkt);
                }
            }
        }
        task.set_timestamp(DONE);
    }

    return PROCESS_OK;
}

void CFFEncoder::set_callback(std::function<CBytes(int64_t,CBytes)> callback_endpoint) {
    callback_endpoint_ = callback_endpoint;
}
