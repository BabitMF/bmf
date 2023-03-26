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

#include "audio_fifo.h"

AudioFifo::AudioFifo(int format, int channels, uint64_t channel_layout, AVRational time_base, int sample_rate) {
    format_ = format;
    channels_ = channels;
    audio_fifo_ = av_audio_fifo_alloc((AVSampleFormat) format, channels, 2048);
    time_base_ = time_base;
    channel_layout_ = channel_layout;
    sample_rate_ = sample_rate;
    pts_per_sample_ = time_base.den / float(time_base.num) / sample_rate;
    if (!audio_fifo_)
        BMFLOG(BMF_ERROR) << "Could not allocate audio_fifo_";
}

int AudioFifo::read(int samples, bool partial, bool &got_frame, AVFrame *&frame) {
    int ret;
    got_frame = false;
    int buffered_samples = av_audio_fifo_size(audio_fifo_);
    if (buffered_samples < 1) {
        return 0;
    }
    if (buffered_samples < samples) {
        if (partial) {
            samples = buffered_samples;
        } else {
            return 0;
        }
    }
    frame->format = format_;
    frame->channel_layout = channel_layout_;
    frame->sample_rate = sample_rate_;
    frame->nb_samples = samples;
    ret = av_frame_get_buffer(frame, 0);
    if (ret < 0) {
        BMFLOG(BMF_ERROR) << "Error allocating an audio buffer";
        return ret;
    }
    int read_samples = av_audio_fifo_read(audio_fifo_, (void **) (frame->extended_data), samples);
    if (read_samples < 0) {
        BMFLOG(BMF_ERROR) << "av_audio_fifo_read " << read_samples;
        return read_samples;
    }
    got_frame = true;
    frame->nb_samples = read_samples;
    if (first_pts_ != AV_NOPTS_VALUE) {
        frame->pts = (int64_t) (pts_per_sample_ * samples_read_) + first_pts_;
    } else {
        frame->pts = AV_NOPTS_VALUE;
    }
    samples_read_ += read_samples;
    return 0;
}

int AudioFifo::read_many(int samples, bool partial, std::vector<AVFrame *> &frame_list) {
    while (1) {
        AVFrame *frame = NULL;
        frame = av_frame_alloc();
        if (!frame) {
            BMFLOG(BMF_ERROR) << "Could not allocate AVFrame";
            return -1;
        }
        bool got_frame = false;
        int ret = read(samples, partial, got_frame, frame);
        if (ret < 0) {
            return ret;
        }
        if (!got_frame) {
            av_frame_free(&frame);
            return 0;
        }
        frame_list.push_back(frame);
    }
    return 0;
}

int AudioFifo::write(AVFrame *frame) {
    int ret;
    if (first_frame_) {
        first_pts_ = frame->pts;
        first_frame_ = false;
    }
    ret = av_audio_fifo_write(audio_fifo_, (void **) (frame->extended_data), frame->nb_samples);
    return ret;
}

AudioFifo::~AudioFifo() {
    if (audio_fifo_)
        av_audio_fifo_free(audio_fifo_);
}
