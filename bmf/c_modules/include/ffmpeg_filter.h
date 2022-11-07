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

#ifndef BMF_FF_FILTER_H
#define BMF_FF_FILTER_H

#include "c_module.h"
#include <bmf/sdk/filter_graph.h>
#include <mutex>

class CFFFilter : public Module {
    std::string g_desc_;
    FilterGraph *filter_graph_;
    std::map<int, AVFilterContext*> buffer_src_ctx_;
    std::map<int, AVFilterContext*> buffer_sink_ctx_;
    std::map<int, std::queue<AVFrame*>> input_cache_;
    std::map<int, std::map<int, std::string>> inpads_;
    std::map<int, std::vector<std::string>> outpads_;
    int node_id_;
    int num_input_streams_;
    int num_output_streams_;
    bool b_graph_inited_;
    bool is_inf_;
    bool all_input_eof_;
    bool all_output_eof_;
    bool copy_ts_;
    std::vector<bool> in_eof_;
    std::vector<bool> out_eof_;
    std::map<int, FilterConfig> config_;
    int64_t stream_start_time_;
    int64_t stream_first_dts_;
    std::mutex reset_check_mutex_;
    std::map<int,int> input_stream_node_;
    std::map<int, std::string> orig_pts_time_cache_;
    JsonParam option_;
public:
    CFFFilter(int node_id, JsonParam option);

    ~CFFFilter();

    int reset();

    int clean();

    int close();

    int parse_filter(std::vector<JsonParam>& f_param, int idx, std::string& result);

    int graph_descr(JsonParam& option, std::string& result);

    int init_filtergraph();

    int process_input_and_output(Task& task, int idx, AVFrame* frm);

    int process(Task &task);

    bool need_hungry_check(int input_stream_id) {return true;};

    bool is_hungry(int input_stream_id);

    bool is_infinity();

    int process_filter_graph(Task &task);

    bmf_sdk::Packet convert_avframe_to_packet(AVFrame* frame, int index);
    bool check_finished();
    int get_cache_frame(int index, AVFrame * &frame, int &choose_index);
};

REGISTER_MODULE_CLASS(CFFFilter);

/** @page ModuleFilter Build-in Filter Module
 * @ingroup FiltM
 * @defgroup FiltM Build-in Filter Module
 */

/** @addtogroup FiltM
 * @{
 * This is a module capability discrption about BMF build-in filter.
 * The module can be used by Module Related BMF API such as bmf.concat() by providing ffmpeg command line style parameters to config the filtergraph:
 * @code
            main_video = (
                video['video'].scale(output_width, output_height)
                    .overlay(logo_1, repeatlast=0)
                    .overlay(logo_2,
                             x='if(gte(t,3.900),960,NAN)',
                             y=0,
                             shortest=1)
            )

            concat_video = (
                bmf.concat(header['video'].scale(output_width, output_height),
                           main_video,
                           tail['video'].scale(output_width, output_height),
                           n=3)
            )

            concat_audio = (
                bmf.concat(header['audio'],
                           video['audio'],
                           tail['audio'],
                           n=3, v=0, a=1)
            )
 * @endcode
 *
 * And in another common way, users can create any filter stream which ffmpeg libavfilter included. exp.:
 * @code
 * ff_filter('unsharp', '5:5:1')
 * @endcode
 * @arg module name: c_ffmpeg_filter\n
 * @}
 */
#endif
