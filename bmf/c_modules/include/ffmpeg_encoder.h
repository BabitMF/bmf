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

#ifndef BMF_FF_ENCODER_H
#define BMF_FF_ENCODER_H

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
#include <libavutil/avstring.h>
};
#include "audio_fifo.h"
#include "audio_resampler.h"
#include "video_sync.h"
#include "av_common_utils.h"
#include <bmf/sdk/filter_graph.h>

typedef struct OutputStream {
    int64_t last_mux_dts;
    int64_t data_size;
    int64_t packets_written;
    int64_t max_frames;
    int64_t frame_number;
    bool encoding_needed;
    std::shared_ptr<AVStream> input_stream;
    int64_t filter_in_rescale_delta_last;
} OutputStream;

typedef struct CurrentImage2Buffer {
    uint8_t *buf;
    size_t size;
    bool is_packing;
    unsigned int room;
} CurrentImage2Buffer;

class CFFEncoder : public Module {
    JsonParam input_option_;
    JsonParam video_params_;
    JsonParam mux_params_;
    JsonParam audio_params_;
    JsonParam metadata_params_;
    AVFormatContext *output_fmt_ctx_;
    AVFrame *decoded_frm_;
    int node_id_;
    int width_;
    int height_;
    int srv_cnt_;
    int fps_;
    int64_t last_pts_;
    std::string output_dir_;
    std::string output_path_;
    std::string output_prefix_;
    std::string codec_names_[2];
    std::string oformat_;
    std::string vtag_;
    std::string atag_;
    AVCodec *codecs_[2];
    int codec_init_touched_num_;
    AVCodecContext *enc_ctxs_[2];
    AVRational in_stream_tbs_[2];
    AVIOContext *avio_ctx_ = NULL;
    int streams_idx_[2];
    struct SwsContext *sws_ctx_;
    struct SwrContext *swr_ctx_;
    AVStream *output_stream_[2];
    int num_input_streams_;
    enum AVPixelFormat pix_fmt_;
    std::vector<std::pair<AVPacket*, int>> cache_;
    std::vector<std::pair<AVPacket*, int>> stream_copy_cache_;
    std::vector<std::pair<AVFrame*,int>> frame_cache_;
    bool stream_inited_;
    bool b_stream_eof_[2];
    bool b_flushed_;
    bool b_eof_;
    bool reset_flag_ = false;
    bool b_init_ = false;
    bool null_output_;
    int push_output_ = 0;
    int push_encoded_output_ = 1;
    int64_t time_ = 0;
    int64_t audio_first_pts_ = LLONG_MAX;
    int64_t video_first_pts_ = LLONG_MAX;
    int64_t first_pts_ = 0;
    int64_t stream_start_time_ = AV_NOPTS_VALUE;
    int64_t stream_first_dts_ = AV_NOPTS_VALUE;
    bool adjust_pts_flag_ = false;
    int n_resample_out_;
    int in_swr_sample_rate_;
    int vsync_method_;
    std::function<CBytes(int64_t,CBytes)> callback_endpoint_;
    std::shared_ptr<VideoSync> video_sync_;
    std::shared_ptr<FilterGraph> output_video_filter_graph_;
    std::shared_ptr<FilterGraph> output_audio_filter_graph_;
    AVRational video_frame_rate_ = {0,0};
    AVRational input_video_frame_rate_ = {0,0};
    AVRational input_sample_aspect_ratio_ = {0,0};
    OutputStream ost_[2];
    int avio_buffer_size_ = 4 * 4096;
    int64_t current_frame_pts_;
    int64_t current_offset_ = 0;
    int current_whence_ = 0;
    Task* current_task_ptr_ = nullptr;
    bool first_packet_[2] = {true, true};
    CurrentImage2Buffer current_image_buffer_ = {0};

public:
    CFFEncoder(int node_id, JsonParam option);

    ~CFFEncoder();

    int init();

    int reset();

    int clean();

    bool check_valid_task(Task& task);

    int handle_output(AVPacket* hpkt, int idx);

    int encode_and_write(AVFrame *frame, unsigned int idx, int *got_frame);

    int init_stream();

    int write_output_data(void *opaque, uint8_t *buf, int buf_size);

    int write_current_packet_data(uint8_t *buf, int buf_size);

    int64_t seek_output_data(void *opaque, int64_t offset, int whence);

    int init_codec(int idx, AVFrame *frame);

    int flush();

    int close();

    int process(Task &task) override;

    bool need_hungry_check(int input_stream_id) {return false;};

    bool is_hungry(int input_stream_id) {return true;};

    bool is_infinity() {return false;};

    int handle_frame(AVFrame* frame,int index);

    int handle_audio_frame(AVFrame *frame, bool is_flushing, int index);

    int handle_video_frame(AVFrame *frame, bool is_flushing, int index);

    void set_callback(std::function<CBytes(int64_t,CBytes)> callback_endpoint) override;

    int get_filter_frame(AVFrame *frame, FilterGraph *fctx, std::vector<AVFrame *> &frames);

    bool need_output_video_filter_graph(AVFrame *frame);

    int streamcopy(AVPacket *ipkt, AVPacket *opkt, int idx);
};

REGISTER_MODULE_CLASS(CFFEncoder)

/** @page ModuleEncoder Build-in Encode Module
 * @ingroup EncM
 * @defgroup EncM Build-in Encode Module
 */

/** @addtogroup EncM
 * @{
 * This is a module capability discrption about BMF build-in encoder.
 * The module can be used by BMF API such as bmf.encode() by providing json style "option" to config such as the 3rd parameter below:
 * @code
            bmf.encode(
                video['video'],
                audio_stream,
                {
                    "output_path": output_path,
                    "video_params": {
                        "codec": "h264",
                        "width": 320,
                        "height": 240,
                        "crf": 23,
                        "preset": "veryfast"
                    },
                    "audio_params": {
                        "codec": "aac",
                        "bit_rate": 128000,
                        "sample_rate": 44100,
                        "channels": 2
                    }
                }
            )
 * @endcode
 * Details:\n
 * @arg module name: c_ffmpeg_encoder\n
 * @}
 */
#endif
