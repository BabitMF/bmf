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

#ifndef C_MODULES_AUDIO_FIFO_H
#define C_MODULES_AUDIO_FIFO_H

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

class AudioFifo {
public:
    AudioFifo(int format, int channels, uint64_t channel_layout, AVRational time_base, int sample_rate);

    ~AudioFifo();

    int write(AVFrame *frame);

    int read(int samples, bool partial, bool &got_frame, AVFrame *&frame);

    int read_many(int samples, bool partial, std::vector<AVFrame *> &frame_list);

    AVAudioFifo *audio_fifo_ = NULL;
    bool first_frame_ = true;
    bool first_pts_ = 0;

    AVRational time_base_;
    int64_t samples_read_ = 0;
    uint64_t channel_layout_ = 0;
    int channels_;
    int format_;
    int sample_rate_;
    float pts_per_sample_ = 0;
};

#endif //C_MODULES_AUDIO_FIFO_H
