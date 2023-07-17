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

#ifndef SIMPLE_FILTER_GRAPH
#define SIMPLE_FILTER_GRAPH

#include <string>
#include <vector>
#include <bmf/sdk/common.h>
#include <bmf/sdk/log.h>
#include <bmf/sdk/filter_graph.h>
#include <map>
#include <memory>

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

struct AVFrame;

BEGIN_BMF_SDK_NS

class FilterGraph;

class SimpleFilterGraph {
  public:
    int init(AVFrame *frame, std::string desc) {
        filter_graph_ = std::make_shared<FilterGraph>();
        FilterConfig fg_config;
        std::map<int, FilterConfig> in_cfgs;
        std::map<int, FilterConfig> out_cfgs;
        fg_config.width = frame->width;
        fg_config.height = frame->height;
        fg_config.format = frame->format;
        fg_config.sample_aspect_ratio = frame->sample_aspect_ratio;
        // get frame rate
        if (frame->metadata) {
            AVDictionaryEntry *tag = NULL;
            while ((tag = av_dict_get(frame->metadata, "", tag,
                                      AV_DICT_IGNORE_SUFFIX))) {
                if (!strcmp(tag->key, "frame_rate")) {
                    std::string svalue = tag->value;
                    int pos = svalue.find(",");
                    if (pos > 0) {
                        AVRational frame_rate;
                        frame_rate.num = stoi(svalue.substr(0, pos));
                        frame_rate.den = stoi(svalue.substr(pos + 1));
                        fg_config.frame_rate = frame_rate;
                    }
                }
                if (!strcmp(tag->key, "time_base")) {
                    std::string svalue = tag->value;
                    int pos = svalue.find(","); //"num,den"
                    if (pos > 0) {
                        AVRational r;
                        r.num = stoi(svalue.substr(0, pos));
                        r.den = stoi(svalue.substr(pos + 1));
                        fg_config.tb = r;
                    }
                }
            }
        }
        in_cfgs[0] = fg_config;
        filter_graph_->config_graph(desc, in_cfgs, out_cfgs);
        return 0;
    };
    int get_filter_frame(AVFrame *frame, std::vector<AVFrame *> &frame_list) {
        AVFrame *av_frame = av_frame_clone(frame);
        filter_graph_->get_filter_frame(av_frame, 0, 0, frame_list);
        av_frame_free(&av_frame);
        return 0;
    };
    std::shared_ptr<FilterGraph> filter_graph_;
};

END_BMF_SDK_NS
#endif // SIMPLE_FILTER_GRAPH
