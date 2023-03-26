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
#include "audio_resampler.h"

AudioResampler::AudioResampler(int input_format, int output_format, int input_channel_layout, int output_channel_layout,
                               int input_sample_rate, int output_sample_rate, AVRational input_time_base,
                               AVRational output_time_base) {
    int ret;
    input_format_ = input_format;
    output_format_ = output_format;
    input_channel_layout_ = input_channel_layout;
    output_channel_layout_ = output_channel_layout;
    input_sample_rate_ = input_sample_rate;
    output_sample_rate_ = output_sample_rate;
    swr_ctx_ = swr_alloc();
    if (!swr_ctx_) {
        BMFLOG(BMF_ERROR) << "Could not allocate swr";
    }
    input_time_base_ = input_time_base;
    output_time_base_ = output_time_base;
    av_opt_set_int(swr_ctx_, "in_sample_fmt", input_format, 0);
    av_opt_set_int(swr_ctx_, "out_sample_fmt", output_format, 0);
    av_opt_set_int(swr_ctx_, "in_channel_layout", input_channel_layout, 0);
    av_opt_set_int(swr_ctx_, "out_channel_layout", output_channel_layout, 0);
    av_opt_set_int(swr_ctx_, "in_sample_rate", input_sample_rate, 0);
    av_opt_set_int(swr_ctx_, "out_sample_rate", output_sample_rate, 0);
    ret = swr_init(swr_ctx_);
    ratio_ = (double) output_sample_rate / input_sample_rate;
    if (ret < 0)
        BMFLOG(BMF_ERROR) << "init swr failed:" << ret;
    return;
}

int AudioResampler::resample(AVFrame *insamples, AVFrame *&outsamples) {
    int ret;
    int n_in = 0;
    if (insamples != NULL) {
        n_in = insamples->nb_samples;
    }
    int n_out = n_in * ratio_;
    int64_t delay;
    delay = swr_get_delay(swr_ctx_, output_sample_rate_);
    if (delay > 0)
        n_out += delay;

    if (insamples != NULL) {
        av_frame_copy_props(outsamples, insamples);
    }
    outsamples->format = output_format_;
    outsamples->channel_layout = output_channel_layout_;
    outsamples->sample_rate = output_sample_rate_;
    outsamples->nb_samples = n_out;

    if (n_out != 0) {
        ret = av_frame_get_buffer(outsamples, 0);
        if (ret < 0) {
            BMFLOG(BMF_ERROR) << "Error allocating an audio buffer";
            return ret;
        }
    } else {
        return 0;
    }


    // get output samples pts
    if (insamples) {
        if (insamples->pts != AV_NOPTS_VALUE && input_time_base_.num != -1) {

            //translate pts to timestamps which is in 1/(in_sample_rate * out_sample_rate) units.
            int64_t inpts = av_rescale(insamples->pts, input_time_base_.num *
                                                       output_sample_rate_ * insamples->sample_rate,
                                       input_time_base_.den);
            // get outpts whose timestamps is 1/(in_sample_rate * out_sample_rate)
            int64_t outpts = swr_next_pts(swr_ctx_, inpts);

            // translate pts to timestamps is output_time_base;
            outsamples->pts = av_rescale(outpts, output_time_base_.den,
                                         output_time_base_.num * output_sample_rate_ * insamples->sample_rate);
        } else {
            outsamples->pts = AV_NOPTS_VALUE;
        }
    } else {
        int64_t outpts = swr_next_pts(swr_ctx_, INT64_MIN);
        outsamples->pts = av_rescale(outpts, output_time_base_.den,
                                     output_time_base_.num * output_sample_rate_ * input_sample_rate_);
    }

    uint8_t **input_data = NULL;
    if (insamples != NULL) {
        input_data = (uint8_t **) insamples->extended_data;
    }
    n_out = swr_convert(swr_ctx_, outsamples->extended_data, n_out, (const uint8_t**)input_data, n_in);
    if (n_out <= 0) {
        return n_out;
    }
    outsamples->nb_samples = n_out;
    return ret;
}

AudioResampler::~AudioResampler() {
    if (swr_ctx_)
        swr_free(&swr_ctx_);
}