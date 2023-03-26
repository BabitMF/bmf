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

#ifndef C_MODULES_VIDEO_SYNC_VFR_H
#define C_MODULES_VIDEO_SYNC_VFR_H

#include "c_module.h"

extern "C" {
#include <libavutil/imgutils.h>
#include <libavutil/samplefmt.h>
#include <libavutil/timestamp.h>
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavutil/imgutils.h>
#include <libavutil/parseutils.h>
#include <libswscale/swscale.h>
#include <libswresample/swresample.h>
#include <libavutil/opt.h>
#include <libavutil/intreadwrite.h>
#include <libavutil/audio_fifo.h>
};

#define VSYNC_AUTO       -1
#define VSYNC_PASSTHROUGH 0
#define VSYNC_CFR         1
#define VSYNC_VFR         2
#define VSYNC_VSCFR       0xfe
#define VSYNC_DROP        0xff

class VideoSync {
public:
    int64_t sync_opts_ = 0;// output frame pts
    AVRational input_stream_time_base_;
    AVRational filter_in_frame_rate_;
    AVRational video_frame_rate_;
    AVRational encode_time_base_;//encode time_base should be the same with frame rate
    int sync_method_;
    AVFrame *last_frame_ = NULL;
    int last_nb0_frames_[3];
    int64_t frame_number_ = 0;
    int64_t stream_start_time_;
    int64_t stream_first_dts_;
    int64_t max_frames_;
    bool last_dropped_ = false;
    int nb_frames_drop_ = 0;
    int nb_frames_dup_ = 0;
    unsigned int dup_warning_ = 1000;
public:
    VideoSync(AVRational input_stream_time_base, AVRational encode_time_base, AVRational filter_in_frame_rate, AVRational video_frame_rate, int64_t stream_start_time, int64_t stream_first_dts, int sync_method, int64_t max_frames);
    ~VideoSync();
    int process_video_frame(AVFrame *frame, std::vector<AVFrame *> &output_frame, int64_t &frame_number);
};

#endif //C_MODULES_VIDEO_SYNC_VFR_H
