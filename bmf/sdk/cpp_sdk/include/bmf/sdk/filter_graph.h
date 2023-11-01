/*
 * Copyright 2023 Babit Authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef C_MODULES_FILTER_GRAPH_H
#define C_MODULES_FILTER_GRAPH_H

#include <bmf/sdk/log.h>
#include <string>
#include <vector>
#include <map>

extern "C" {
#include <libavutil/imgutils.h>
#include <libavutil/samplefmt.h>
#include <libavutil/timestamp.h>
#include <libavformat/avformat.h>
#include <libavfilter/avfilter.h>
#include <libavfilter/buffersink.h>
#include <libavfilter/buffersrc.h>
#include <libavcodec/avcodec.h>
#include <libavutil/imgutils.h>
#include <libavutil/parseutils.h>
#include <libavutil/bprint.h>
#include <libswscale/swscale.h>
#include <libavutil/opt.h>
#include <libavutil/error.h>
};

BEGIN_BMF_SDK_NS

class FilterConfig {
  public:
    int format;
    int width, height;
    AVRational sample_aspect_ratio;
    int sample_rate;
    int channels;
    uint64_t channel_layout;
    AVRational tb = AVRational{1, 25};
    AVRational frame_rate = AVRational{0, 1};
    unsigned frame_size = 0;
};

class FilterGraph {
    AVFilterInOut *outputs_;
    AVFilterInOut *inputs_;
    std::string graph_desc_;
    std::map<int, FilterConfig> in_configs_;
    std::map<int, FilterConfig> out_configs_;
    bool b_init_;

  public:
    AVFilterGraph *filter_graph_;
    std::map<int, AVBufferRef *> hw_frames_ctx_map_;
    std::map<int, AVFilterContext *> buffer_src_ctx_;
    std::map<int, AVFilterContext *> buffer_sink_ctx_;

    FilterGraph() { init(); };

    ~FilterGraph() { clean(); };

    int init() {
        outputs_ = NULL;
        inputs_ = NULL;
        filter_graph_ = avfilter_graph_alloc();
        b_init_ = true;
        if (!filter_graph_) {
            BMFLOG(BMF_ERROR) << "Graph alloc error: ENOMEM";
            b_init_ = false;
            return -1;
        }

        /** @addtogroup FiltM
         * @{
         * @env BMF_FILTERGRAPH_THREADS: set nb_threads of ffmpeg filter_graph,
         * for example, BMF_FILTERGRAPH_THREADS=1
         * @} */
        char *threads_env = getenv("BMF_FILTERGRAPH_THREADS");
        if (threads_env) {
            std::string threads_str = threads_env;
            BMFLOG(BMF_DEBUG) << "env BMF_FILTERGRAPH_THREADS: " << threads_str;
            filter_graph_->nb_threads = std::stoi(threads_str);
        }
        return 0;
    };

    int clean() {
        if (filter_graph_)
            avfilter_graph_free(&filter_graph_);
        if (inputs_)
            avfilter_inout_free(&inputs_);
        if (outputs_)
            avfilter_inout_free(&outputs_);
        for (auto it : hw_frames_ctx_map_) {
            if (it.second) {
                av_buffer_unref(&it.second);
            }
        }
        hw_frames_ctx_map_.clear();

        b_init_ = false;

        return 0;
    };

    bool check_hw_device_ctx_uniformity() {
        if (hw_frames_ctx_map_.size() == 0)
            return false;

        auto it = hw_frames_ctx_map_.begin();

        auto base_frames_ctx = (AVHWFramesContext *)it->second->data;
        auto base_device_ctx = base_frames_ctx->device_ctx;

        ++it;

        for (; it != hw_frames_ctx_map_.end(); ++it) {
            if (it->second) {
                AVHWFramesContext *frame_ctx =
                    (AVHWFramesContext *)it->second->data;
                AVHWDeviceContext *device_ctx = frame_ctx->device_ctx;
                if (device_ctx != base_device_ctx) {
                    return false;
                }
            }
        }

        return true;
    }

    int config_graph(std::string &graph_desc,
                     std::map<int, FilterConfig> &config,
                     std::map<int, FilterConfig> &out_config) {

        // prepend sws_flags if required, apply global sws flag only for scale
        // or format filters
        std::string flag_prefix = "sws_flags=";
        if (getenv("BMF_SWS_FLAGS") &&
            graph_desc.substr(0, flag_prefix.size()) != flag_prefix) {
            bool use_global_flag =
                graph_desc.find("scale=") != std::string::npos;
            if (!use_global_flag) {
                size_t format_pos = graph_desc.find("format=", 0);
                while (format_pos != std::string::npos && !use_global_flag) {
                    if (format_pos > 0 && graph_desc[format_pos - 1] != 'a')
                        use_global_flag = true;
                    format_pos = graph_desc.find("format=", format_pos + 1);
                }
            }
            if (use_global_flag)
                graph_desc =
                    flag_prefix + getenv("BMF_SWS_FLAGS") + ";" + graph_desc;
        }

        graph_desc_ = graph_desc;
        in_configs_ = config;
        out_configs_ = out_config;

        // [i0_0] [i1_0] [i2_0] [i3_0] ... [inputStream_Pin]
        // [o0_0] [o1_0] .... [outputStream_Pin]
        AVBPrint args;
        AVBPrint out_args;
        int ret = 0;
        AVFilterInOut *curr;
        enum AVPixelFormat pix_fmts[] = {AV_PIX_FMT_YUV420P, AV_PIX_FMT_NONE};

        if (!b_init_) {
            BMFLOG(BMF_DEBUG) << "FilterGraph is not inited";
            return -1;
        }
        BMFLOG(BMF_DEBUG) << "graph desc: " << graph_desc;
        if ((ret = avfilter_graph_parse2(filter_graph_, graph_desc.c_str(),
                                         &inputs_, &outputs_)) < 0) {
            BMFLOG(BMF_ERROR) << "Graph parse2 error: " << graph_desc;
            goto end;
        }

        // set hw device context of AVFilterContext
        if (check_hw_device_ctx_uniformity()) {
            auto it = hw_frames_ctx_map_.begin();
            auto base_frames_ctx = (AVHWFramesContext *)it->second->data;
            AVBufferRef *base_device_ctx_buf = base_frames_ctx->device_ref;
            for (int i = 0; i < filter_graph_->nb_filters; i++) {
                filter_graph_->filters[i]->hw_device_ctx =
                    av_buffer_ref(base_device_ctx_buf);
            }
        }

        curr = inputs_;
        while (curr) {
            const AVFilter *buffersrc;
            // const AVFilter *fifo;
            std::string stream_str = curr->name;
            int start_pos = stream_str.find("i");
            int end_pos = stream_str.find("_");
            int st = std::stoi(stream_str.substr(start_pos + 1, end_pos));
            if (st < 0 || st >= config.size())
                BMFLOG(BMF_WARNING)
                    << "Graph config missed for filter index " << st;

            enum AVMediaType type = avfilter_pad_get_type(
                curr->filter_ctx->input_pads, curr->pad_idx);

            FilterConfig input_config = {0};
            if (config.find(st) == config.end()) {
                BMFLOG(BMF_WARNING)
                    << "Graph input filter config not set by stream, might be "
                       "a empty stream, will try to set by default"
                    << st;
                if (type == AVMEDIA_TYPE_VIDEO) {
                    input_config.width = 1280;
                    input_config.height = 720;
                    input_config.format = 0;
                    input_config.tb = AVRational{1, 1000000};
                    input_config.frame_rate = AVRational{30, 1};
                } else {
                    input_config.tb = AVRational{1, 44100};
                    input_config.sample_rate = 44100;
                    input_config.channel_layout = 0x3;
                    input_config.channels = 2;
                    input_config.format = 8;
                }
                config[st] = input_config;
            } else
                input_config = config[st];

            // time_base = type == AVMEDIA_TYPE_VIDEO ? (AVRational){1, 25} :
            // (AVRational){1, input_cache->sample_rate};
            av_bprint_init(&args, 0, AV_BPRINT_SIZE_AUTOMATIC);
            if (type == AVMEDIA_TYPE_VIDEO) {
                buffersrc = avfilter_get_by_name("buffer");
                // fifo = avfilter_get_by_name("fifo");

                av_bprintf(&args, "video_size=%dx%d:pix_fmt=%d",
                           input_config.width, input_config.height,
                           input_config.format);
                if (input_config.tb.num != 0 && input_config.tb.den != 0)
                    av_bprintf(&args, ":time_base=%d/%d", input_config.tb.num,
                               input_config.tb.den);
                if (input_config.sample_aspect_ratio.num != 0 &&
                    input_config.sample_aspect_ratio.den != 0)
                    av_bprintf(&args, ":pixel_aspect=%d/%d",
                               input_config.sample_aspect_ratio.num,
                               input_config.sample_aspect_ratio.den);
                if (input_config.frame_rate.num != 0)
                    av_bprintf(&args, ":frame_rate=%d/%d",
                               input_config.frame_rate.num,
                               input_config.frame_rate.den);

                BMFLOG(BMF_DEBUG) << "ffmpeg_filter video args: " << args.str;
            } else {
                buffersrc = avfilter_get_by_name("abuffer");

                if (input_config.channel_layout == 0) {
                    if (input_config.channels)
                        input_config.channel_layout =
                            av_get_default_channel_layout(
                                input_config.channels);
                }

                av_bprintf(&args,
                           "time_base=%d/"
                           "%d:sample_rate=%d:sample_fmt=%s:channel_"
                           "layout=0x%" PRIx64,
                           input_config.tb.num, input_config.tb.den,
                           input_config.sample_rate,
                           av_get_sample_fmt_name(
                               (enum AVSampleFormat)input_config.format),
                           input_config.channel_layout);
                BMFLOG(BMF_DEBUG) << "ffmpeg_filter audio args: " << args.str;
            }

            AVFilterContext *buffersrc_ctx;
            std::string fname = "src_" + std::to_string(st);
            ret = avfilter_graph_create_filter(&buffersrc_ctx, buffersrc,
                                               fname.c_str(), args.str, NULL,
                                               filter_graph_);
            av_bprint_finalize(&args, NULL);
            if (type == AVMEDIA_TYPE_VIDEO) {
                if (hw_frames_ctx_map_.find(st) != hw_frames_ctx_map_.end()) {
                    AVBufferSrcParameters *par =
                        av_buffersrc_parameters_alloc();
                    memset(par, 0, sizeof(*par));
                    par->format = AV_PIX_FMT_NONE;
                    par->hw_frames_ctx = hw_frames_ctx_map_[st];
                    av_buffersrc_parameters_set(buffersrc_ctx, par);
                }
            }

            if (ret < 0) {
                BMFLOG(BMF_ERROR) << "Cannot create buffer source";
                goto end;
            }

            ret = avfilter_link(buffersrc_ctx, 0, curr->filter_ctx,
                                curr->pad_idx);
            if (ret < 0) {
                BMFLOG(BMF_ERROR) << "Link error";
                goto end;
            }
            buffer_src_ctx_[st] = buffersrc_ctx;

            curr = curr->next;
        }

        curr = outputs_;
        while (curr) {
            const AVFilter *format;
            const AVFilter *buffersink;
            AVFilterContext *format_ctx = NULL;
            std::string stream_str = curr->name;
            int start_pos = stream_str.find("o");
            int end_pos = stream_str.find("_");
            int st = std::stoi(stream_str.substr(start_pos + 1, end_pos));

            enum AVMediaType type = avfilter_pad_get_type(
                curr->filter_ctx->output_pads, curr->pad_idx);

            char *out_arg_str = NULL;
            if (type == AVMEDIA_TYPE_AUDIO) {
                FilterConfig output_config;
                if (out_config.find(st) != out_config.end()) {
                    output_config = out_config[st];
                    if (output_config.sample_rate != 0 &&
                        output_config.channel_layout != 0) {
                        av_bprint_init(&out_args, 0, AV_BPRINT_SIZE_AUTOMATIC);
                        format = avfilter_get_by_name("aformat");
                        av_bprintf(
                            &out_args,
                            "sample_rates=%d:sample_fmts=%s:channel_"
                            "layouts=0x%" PRIx64,
                            output_config.sample_rate,
                            av_get_sample_fmt_name(
                                (enum AVSampleFormat)output_config.format),
                            output_config.channel_layout);
                        out_arg_str = out_args.str;
                        BMFLOG(BMF_DEBUG)
                            << "ffmpeg_filter buffer sink for audio args: "
                            << out_arg_str;

                        std::string fname = "format_" + std::to_string(st);
                        ret = avfilter_graph_create_filter(
                            &format_ctx, format, fname.c_str(), out_arg_str,
                            NULL, filter_graph_);
                        av_bprint_finalize(&out_args, NULL);
                        if (ret < 0) {
                            BMFLOG(BMF_ERROR) << "Cannot create format filter";
                            goto end;
                        }

                        ret = avfilter_link(curr->filter_ctx, curr->pad_idx,
                                            format_ctx, 0);
                        if (ret < 0) {
                            BMFLOG(BMF_ERROR) << "Link error";
                            goto end;
                        }
                    }
                }
            }

            if (type == AVMEDIA_TYPE_VIDEO) {
                buffersink = avfilter_get_by_name("buffersink");
            } else {
                buffersink = avfilter_get_by_name("abuffersink");
            }

            AVFilterContext *buffersink_ctx;
            std::string fname = "sink_" + std::to_string(st);
            ret = avfilter_graph_create_filter(&buffersink_ctx, buffersink,
                                               fname.c_str(), NULL, NULL,
                                               filter_graph_);
            if (ret < 0) {
                BMFLOG(BMF_ERROR) << "Cannot create buffer sink";
                goto end;
            }

            AVFilterContext *link_ctx =
                format_ctx ? format_ctx : curr->filter_ctx;
            ret = avfilter_link(link_ctx, curr->pad_idx, buffersink_ctx, 0);
            if (ret < 0) {
                BMFLOG(BMF_ERROR) << "Link error";
                goto end;
            }
            buffer_sink_ctx_[st] = buffersink_ctx;

            curr = curr->next;
        }

        if ((ret = avfilter_graph_config(filter_graph_, NULL)) < 0) {
            BMFLOG(BMF_ERROR) << "Graph config error";
            goto end;
        }

        for (auto &out_config : out_configs_) {
            if (out_config.second.frame_size != 0) {
                av_buffersink_set_frame_size(buffer_sink_ctx_[out_config.first],
                                             out_config.second.frame_size);
            }
        }

    end:
        avfilter_inout_free(&inputs_);
        avfilter_inout_free(&outputs_);

        return ret;
    };

    int get_filter_frame(AVFrame *frame, int in_idx, int out_idx,
                         std::vector<AVFrame *> &frames) {
        int ret;
        AVFrame *filt_frame;

        if (check_input_property(frame, in_idx) < 0) {
            BMFLOG(BMF_ERROR) << "Graph check input property failed";
            return -1;
        }

        if (buffer_src_ctx_.find(in_idx) == buffer_src_ctx_.end()) {
            BMFLOG(BMF_ERROR) << "Buffer src" << in_idx << " cann not be found";
            return -1;
        }
        if (buffer_sink_ctx_.find(out_idx) == buffer_sink_ctx_.end()) {
            BMFLOG(BMF_ERROR)
                << "Buffer out" << out_idx << " cann not be found";
            return -1;
        }

        ret = av_buffersrc_add_frame_flags(buffer_src_ctx_[in_idx], frame,
                                           AV_BUFFERSRC_FLAG_PUSH);
        if (ret < 0) {
            av_log(NULL, AV_LOG_ERROR, "Error while feeding the filtergraph\n");
            return ret;
        }

        while (1) {
            ret = avfilter_graph_request_oldest(filter_graph_);
            if (ret < 0 && ret != AVERROR_EOF && ret != AVERROR(EAGAIN)) {
                BMFLOG(BMF_ERROR) << "request oldest from graph error";
                return ret;
            }
            filt_frame = av_frame_alloc();
            if (!filt_frame) {
                ret = AVERROR(ENOMEM);
                break;
            }
            ret = av_buffersink_get_frame_flags(buffer_sink_ctx_[out_idx],
                                                filt_frame,
                                                AV_BUFFERSINK_FLAG_NO_REQUEST);
            if (ret < 0) {
                /* if no more frames for output - returns AVERROR(EAGAIN)
                 * if flushed and no more frames for output - returns
                 * AVERROR_EOF rewrite retcode to 0 to show it as normal
                 * procedure completion
                 */
                if (ret == AVERROR(EAGAIN))
                    ret = 0;
                av_frame_free(&filt_frame);
                break;
            }
            frames.push_back(filt_frame);
        }

        return ret;
    };

    int check_input_property(AVFrame *frame, int in_idx) {
        bool need_reset = false;
        FilterConfig *config =
            in_configs_.count(in_idx) > 0 ? &in_configs_[in_idx] : NULL;

        if (!config || !frame)
            return 0;
        if (frame->width && frame->height) {
            if (frame->width != config->width ||
                frame->height != config->height) {
                config->width = frame->width;
                config->height = frame->height;
                need_reset = true;
            }
        } else {
            if (frame->sample_rate != config->sample_rate ||
                frame->channels != config->channels ||
                frame->channel_layout != config->channel_layout) {
                config->sample_rate = frame->sample_rate;
                config->channels = frame->channels;
                config->channel_layout = frame->channel_layout;
                config->tb = av_make_q(1, config->sample_rate);
                if (frame->metadata) {
                    AVDictionaryEntry *tag = NULL;
                    while ((tag = av_dict_get(frame->metadata, "", tag,
                                              AV_DICT_IGNORE_SUFFIX))) {
                        if (!strcmp(tag->key, "time_base")) {
                            std::string svalue = tag->value;
                            int pos = svalue.find(",");
                            if (pos > 0) {
                                AVRational r;
                                r.num = stoi(svalue.substr(0, pos));
                                r.den = stoi(svalue.substr(pos + 1));
                                config->tb = r;
                            }
                        }
                    }
                }
                need_reset = true;
            }
        }
        if (need_reset) {
            clean();
            if (init() != 0)
                return -1;
            if (config_graph(graph_desc_, in_configs_, out_configs_) != 0)
                return -1;
            BMFLOG(BMF_INFO) << "Graph: " << graph_desc_ << " has been reset";
            return 1;
        }
        return 0;
    };

    int get_the_most_failed_nb_request() {
        int index = -1;
        int most_failed_request = 0;
        for (int i = 0; i < buffer_src_ctx_.size(); i++) {
            int failed_request =
                av_buffersrc_get_nb_failed_requests(buffer_src_ctx_[i]);
            if (failed_request > most_failed_request) {
                most_failed_request = failed_request;
                index = i;
            }
        }
        return index;
    };

    int reap_filters(std::map<int, std::vector<AVFrame *>> &output_frames,
                     int mode) {
        int ret = 0;
        if (mode == 0) {
            int ret = avfilter_graph_request_oldest(filter_graph_);
            if (ret < 0 && ret != AVERROR_EOF) {
                return ret;
            }
        }
        for (int i = 0; i < buffer_sink_ctx_.size(); i++) {
            while (1) {
                AVFrame *frame = av_frame_alloc();
                int index = i;
                int ret = av_buffersink_get_frame_flags(
                    buffer_sink_ctx_[i], frame, AV_BUFFERSINK_FLAG_NO_REQUEST);
                if (ret < 0) {
                    av_frame_free(&frame);
                    if (ret == AVERROR_EOF) {
                        output_frames[index].push_back(NULL);
                    } else if (ret == AVERROR(EAGAIN)) {
                        break;
                    } else {
                        return ret;
                    }
                    break;
                }
                static int sink_get_frame = 0;
                sink_get_frame++;

                static int video_frame_num = 0;
                static int audio_frame_num = 0;
                if (index == 0) {
                    video_frame_num++;
                } else {
                    audio_frame_num++;
                }

                output_frames[index].push_back(frame);
            }
        }
        return 0;
    };

    int push_frame(AVFrame *frame, int index) {
        int ret = 0;
        if (ret = check_input_property(frame, index) >= 0) {
            ret = av_buffersrc_add_frame_flags(buffer_src_ctx_[index], frame,
                                               AV_BUFFERSRC_FLAG_PUSH |
                                                   AV_BUFFERSRC_FLAG_KEEP_REF);
            if (ret < 0) {
                if (ret != AVERROR_EOF) {
                    return ret;
                }
            }
        } else {
            return ret;
        }
        return 0;
    };
};

END_BMF_SDK_NS

#endif // C_MODULES_FILTER_GRAPH_H
