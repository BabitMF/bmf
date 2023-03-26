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

#ifndef C_MODULES_AUDIO_RESAMPLER_H
#define C_MODULES_AUDIO_RESAMPLER_H
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
};

class AudioResampler {
public:

    AudioResampler(int input_format, int output_format, int input_channel_layout, int output_channel_layout,
                   int input_sample_rate, int output_sample_rate, AVRational input_time_base,
                   AVRational output_time_base);

    ~AudioResampler();

    int resample(AVFrame *insamples, AVFrame *&result);

    struct SwrContext *swr_ctx_;
    int input_format_;
    int output_format_;
    int input_channel_layout_;
    int output_channel_layout_;
    int input_sample_rate_;
    int output_sample_rate_;
    AVRational input_time_base_;
    AVRational output_time_base_;
    double ratio_ = 0;
};

#endif //C_MODULES_AUDIO_RESAMPLER_H
