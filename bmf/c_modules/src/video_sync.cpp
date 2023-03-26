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

#include"video_sync.h"

static inline av_const int mid_pred(int a, int b, int c)
{
    if(a>b) {
        if(c>b) {
            if(c>a) b=a;
            else    b=c;
        }
    } else {
        if(b>c) {
            if(c>a) b=c;
            else    b=a;
        }
    }
    return b;
}

VideoSync::VideoSync(AVRational input_stream_time_base, AVRational encode_time_base, AVRational filter_in_frame_rate, AVRational video_frame_rate, int64_t stream_start_time, int64_t stream_first_dts, int sync_method, int64_t max_frames)
{
    input_stream_time_base_ = input_stream_time_base;
    encode_time_base_ = encode_time_base;
    filter_in_frame_rate_ = filter_in_frame_rate;
    video_frame_rate_ = video_frame_rate;
    sync_method_ = sync_method;
    memset(last_nb0_frames_, 0, sizeof(int));
    stream_start_time_ = stream_start_time;
    stream_first_dts_ = stream_first_dts;
    max_frames_ = max_frames > 0 ? max_frames : INT64_MAX;
}

VideoSync::~VideoSync()
{
    if (nb_frames_drop_ > 0)
        BMFLOG(BMF_INFO) << "drop frames:" << nb_frames_drop_;
    if (nb_frames_dup_ > 0)
        BMFLOG(BMF_INFO) << "dup frames:" << nb_frames_dup_;
}

int VideoSync::process_video_frame(AVFrame *frame, std::vector<AVFrame *> &output_frame, int64_t &frame_number)
{
    //translate to float pts
    double float_pts = 0;
    AVRational temp_encode_time_base = encode_time_base_;
    int extra_bits = 0;
    double duration = 0;
    int nb_frames, nb0_frames;
    double delta0, delta;

    if (!frame) {
        nb0_frames = nb_frames = mid_pred(last_nb0_frames_[0], last_nb0_frames_[1], last_nb0_frames_[2]);
    } else {
        nb0_frames = 0;
        nb_frames = 1;
        extra_bits = av_clip(29 - av_log2(temp_encode_time_base.den), 0, 16);
        temp_encode_time_base.den <<= extra_bits;
        float_pts = av_rescale_q(frame->pts, input_stream_time_base_, temp_encode_time_base);
                    //- (ost_start_time_ != AV_NOPTS_VALUE ?
                    //   av_rescale_q(ost_start_time_, AV_TIME_BASE_Q, temp_encode_time_base) : 0);
        float_pts /= 1 << extra_bits;
        float_pts += FFSIGN(float_pts) * 1.0 / (1<<17);

        if (filter_in_frame_rate_.num > 0 && filter_in_frame_rate_.den > 0) {
            duration = 1 / (av_q2d(filter_in_frame_rate_) * av_q2d(encode_time_base_));
        }

        if (stream_start_time_ != AV_NOPTS_VALUE && stream_first_dts_ != AV_NOPTS_VALUE && video_frame_rate_.num)
            duration = FFMIN(duration, 1/(av_q2d(video_frame_rate_) * av_q2d(encode_time_base_)));

        if (filter_in_frame_rate_.num == 0 && filter_in_frame_rate_.den == 0
            && lrint(frame->pkt_duration * av_q2d(input_stream_time_base_) / av_q2d(encode_time_base_)) > 0)
            duration = lrintf(frame->pkt_duration * av_q2d(input_stream_time_base_) / av_q2d(encode_time_base_));

        delta0 = float_pts - sync_opts_;
        delta = delta0 + duration;

        if (delta0 < 0 && delta > 0 && sync_method_ != VSYNC_PASSTHROUGH &&
            sync_method_ != VSYNC_DROP) {
            if (delta0 < -0.6) {
                BMFLOG(BMF_INFO) << "Past duration" << -delta0 << " too large";
            } else
                BMFLOG(BMF_DEBUG) << "Clipping frame in rate conversion by " << -delta0;
            float_pts = sync_opts_;
            duration += delta0;
            delta0 = 0;
        }

        switch (sync_method_) {
        //case VSYNC_VSCFR:
        //    if (ost->frame_number == 0 && delta0 >= 0.5) {
        //        av_log(NULL, AV_LOG_DEBUG, "Not duplicating %d initial frames\n", (int)lrintf(delta0));
        //        delta = duration;
        //        delta0 = 0;
        //        ost->sync_opts = lrint(sync_ipts);
        //    }
        case VSYNC_CFR:
            //if (frame_drop_threshold && delta < frame_drop_threshold && ost->frame_number)
            if (delta < -1.1)
                nb_frames = 0;
            else if (delta > 1.1) {
                nb_frames = lrintf(delta);
                if (delta0 > 1.1)
                    nb0_frames = lrintf(delta0 - 0.6);
            }
            break;
        case VSYNC_VFR:
            if (delta <= -0.6)
                nb_frames = 0;
            else if (delta > 0.6)
                sync_opts_ = lrint(float_pts);
            break;
        case VSYNC_DROP:
        case VSYNC_PASSTHROUGH:
            sync_opts_ = lrint(float_pts);
            break;
        default:
            break;
        }
    }

    nb_frames = FFMIN(nb_frames, max_frames_ - frame_number);
    nb0_frames = FFMIN(nb0_frames, nb_frames);

    memmove(last_nb0_frames_ + 1,
            last_nb0_frames_,
            sizeof(last_nb0_frames_[0]) * (FF_ARRAY_ELEMS(last_nb0_frames_) - 1));
    last_nb0_frames_[0] = nb0_frames;

    if (nb0_frames == 0 && last_dropped_) {
        nb_frames_drop_++;
        BMFLOG(BMF_INFO) << "*** dropping frame " << frame_number_ << " at ts " << last_frame_->pts;
    }
    if (nb_frames > (nb0_frames && last_dropped_) + (nb_frames > nb0_frames)) {
        float dts_error_threshold   = 3600*30;
        if (nb_frames > dts_error_threshold * 30) {
            av_log(NULL, AV_LOG_ERROR, "%d frame duplication too large, skipping\n", nb_frames - 1);
            nb_frames_drop_++;
            return -1;
        }
        nb_frames_dup_ += nb_frames - (nb0_frames && last_dropped_) - (nb_frames > nb0_frames);
        BMFLOG(BMF_INFO) << "*** " << nb_frames -1 << " dup!";
        if (nb_frames_dup_ > dup_warning_) {
            BMFLOG(BMF_WARNING) << "More than " << dup_warning_ << " frames duplicated";
            dup_warning_ *= 10;
        }
    }

    last_dropped_ = nb_frames == nb0_frames && frame;

    for (int i = 0; i < nb_frames; i++) {
        AVFrame *in_picture;

        if (i < nb0_frames && last_frame_) {
            in_picture = av_frame_clone(last_frame_);
        }
        else
            in_picture = av_frame_clone(frame);

        in_picture->pts = sync_opts_;
        output_frame.push_back(in_picture);
        sync_opts_++;
        frame_number_++;
    }

    if (!last_frame_)
        last_frame_ = av_frame_alloc();
    av_frame_unref(last_frame_);
    if (frame && last_frame_) {
        av_frame_ref(last_frame_, frame);
    } else
        av_frame_free(&last_frame_);

    return 0;
}
