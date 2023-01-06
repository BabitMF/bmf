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

#ifndef BMF_FF_DECODER_H
#define BMF_FF_DECODER_H

#include <bmf/sdk/bmf_av_packet.h>
#include <bmf/sdk/filter_graph.h>
#include "c_module.h"
#include "video_sync.h"
#include <condition_variable>
#include <thread>
#include "av_common_utils.h"

extern "C" {
#include <libavutil/imgutils.h>
#include <libavutil/samplefmt.h>
#include <libavutil/timestamp.h>
#include <libavformat/avformat.h>
#include <libavfilter/avfilter.h>
#include <libavfilter/buffersink.h>
#include <libavfilter/buffersrc.h>
#include <libavutil/hwcontext.h>
};

#ifdef BMF_USE_MEDIACODEC
extern "C" 
{
    #include "libavcodec/jni.h"
}
#include "c_android_vm.h"
#endif

typedef struct FilteringContext {
    AVFilterContext *buffersink_ctx;
    AVFilterContext *buffersrc_ctx;
    AVFilterGraph *filter_graph;
} FilteringContext;

typedef struct InputStream {
    int64_t next_dts;
    int64_t next_pts;
    int64_t max_pts;
    int64_t min_pts;
    int64_t pts;
    int64_t dts;
    int64_t filter_in_rescale_delta_last;
    int64_t max_frames;
    int64_t frame_number;
    int saw_first_ts;
    int wrap_correction_done;
    int64_t sample_decoded;
    int64_t frame_decoded;
    int64_t *dts_buffer;
    int nb_dts_buffer;
    bool decoding_needed;
    bool codecpar_sended;
} InputStream;

class CFFDecoder : public Module {
    JsonParam dec_params_;
    AVInputFormat *input_format_;
    AVFormatContext *input_fmt_ctx_;
    AVFrame *decoded_frm_;
    int node_id_;
    std::string input_path_;
    JsonParam jparam_;
    int video_stream_index_;
    int audio_stream_index_;
    AVCodecContext *video_decode_ctx_;
    AVCodecContext *audio_decode_ctx_;
    AVStream *video_stream_;
    AVStream *audio_stream_;
    int video_frame_count_;
    int audio_frame_count_;
    int refcount_;
    int fps_;
    int64_t start_time_;
    int64_t end_time_;
    int64_t end_video_time_;
    int64_t end_audio_time_;
    int64_t last_pts_;
    int64_t curr_pts_;
    int64_t last_ts_;
    int64_t ts_offset_;
    std::vector<double> durations_;
    int idx_dur_;
    bool dur_end_[2];
    AVRational video_time_base_;
    AVRational audio_time_base_;
    enum AVDiscard skip_frame_;
    bool video_end_;
    bool audio_end_;
    bool encrypted_;
    enum AVPixelFormat pix_fmt_;
    FilteringContext fctx_;
    bool auto_rotate_flag_ = true;
    bool init_filter_ctx_ = false;
    std::queue<BMFAVPacket> bmf_av_packet_queue_;
    std::queue<int64_t> input_timestamp_queue_;
    std::string video_time_base_string_;
    std::string audio_time_base_string_;
    std::string video_codec_name_;
    std::string audio_codec_name_;
    AVIOContext *avio_ctx_ = NULL;
    int64_t decoded_pts_ = 0;
    int current_packet_loc_ = 0;
    bool push_data_flag_ = false;
    bool init_done_ = false;
    int input_type_ = VIDEO_TYPE;
    bool next_file_ = false;
    bool has_input_ = false;
    bool fg_inited_[2] = {false,false};
    std::vector<JsonParam> file_list_;
    bool task_done_ = false;
    std::map<int, FilterConfig> config_;
    FilterGraph *filter_graph_[2];
    InputStream ist_[2];
    std::string hwaccel_str_;
    int hwaccel_check_ = 0;
    AVDictionary *dec_opts_ = NULL;
    bool copy_ts_ = false;

    //use when input is bmf_avpacket
    bool packets_handle_all_ = false;
    bool valid_packet_flag_ = false;
    std::condition_variable process_var_;
    std::mutex read_packet_mutex_;
    std::condition_variable packet_ready_;
    std::mutex process_mutex_;
    std::thread exec_thread_;
    Task task_;
    double extract_frames_fps_ = 0;
    std::string extract_frames_device_;
    std::shared_ptr<VideoSync> video_sync_ = NULL;
    bool start_decode_flag_ = false;
    bool handle_input_av_packet_flag_ = false;
    bool input_eof_packet_received_ = false;
    bool first_handle_ = true;
    int64_t adjust_pts_ = 0;
    int64_t last_output_pts_ = 0;
    int64_t first_video_start_time_ = -1;
    int64_t temp_last_output_pts_ = 0;
    int64_t overlap_time_;
    int64_t cut_off_time_;
    int64_t cut_off_interval_;
    bool stream_copy_av_stream_flag_[2];
    float max_error_rate_ = 2.0/3;
    int64_t decode_error_[2] = {0};
    int64_t stream_frame_number_ = 0;
    std::mutex mutex_;

    //for raw stream input
    int push_raw_stream_;
    int push_audio_channels_;
    int push_audio_sample_rate_;
    int push_audio_sample_fmt_;

    int orig_pts_time_ = 0;

    int process_raw_stream_packet(Task &task, BMFAVPacket &bmf_pkt, bool eof);

    int process_input_bmf_av_packet(Task &task);
    int mv_task_data(Task &dst_task);
    int start_decode(std::vector<int> input_index, std::vector<int> output_index);
    int codec_context(int *stream_idx,
                      AVCodecContext **dec_ctx, AVFormatContext *fmt_ctx, enum AVMediaType type);

    int init_input(AVDictionary *);

    int decode_send_packet(Task &task, AVPacket *pkt, int *got_frame);

    bool check_valid_packet(AVPacket *pkt, Task &task);

    int handle_output_data(Task &task, int type, AVPacket *pkt, bool eof, bool repeat, int got_output);

    Packet generate_video_packet(AVFrame *frame);
    Packet generate_audio_packet(AVFrame *frame);
    int process_task_output_packet(int index, Packet &packet);
    int64_t get_start_time();
    int extract_frames(AVFrame *frame, std::vector<AVFrame*> &output_frames);

#ifdef BMF_USE_MEDIACODEC
    int init_android_vm();
#endif

public:
    CFFDecoder(int node_id, JsonParam option);

    ~CFFDecoder();

    int reset() { return 0; };

    int close();

    int clean();

    int flush(Task &task);

    int process(Task &task);

    int get_rotate_desc(std::string &filter_desc);

    bool need_hungry_check(int input_stream_id) { return false; };

    bool is_hungry(int input_stream_id) { return true; };

    bool is_infinity() { return false; };

    int init_codec(AVPacket *pkt);

    int init_av_codec();
    int init_packet_av_codec();

    int init_filtergraph(int index, AVFrame *frame);

    int read_packet(uint8_t *buf, int buf_size);

    int pkt_ts(AVPacket *pkt, int index);
};
/** @page ModuleDecoder Build-in Decode Module
 * @ingroup DecM
 * @defgroup DecM Build-in Decode Module
 */

/** @addtogroup DecM
 * @{
 * This is a module capability discrption about BMF build-in decoder.
 * The module can be used by BMF API such as bmf.decode() by providing json style "option" to config such as the 3rd parameter below:
 * @code
            bmf.decode(
                {'input_path': input_video_path}
            )
 * @endcode
 * Details:\n
 * @arg module name: c_ffmpeg_decoder\n
 * @}
 */

#endif //BMF_FF_DECODER_H
