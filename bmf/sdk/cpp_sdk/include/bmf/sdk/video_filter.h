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
#pragma once

#include "common.h"
#include <bmf/sdk/common.h>
#include <bmf/sdk/video_frame.h>
#include <bmf/sdk/json_param.h>

#include <unordered_map>
#include <functional>

BEGIN_BMF_SDK_NS

using VFFilter = std::function<VideoFrame(VideoFrame &, JsonParam)>;

class VFFilterManager {
  public:
    static VFFilterManager &get_instance() {
        static VFFilterManager instance;
        return instance;
    }

    void register_filter(const std::string &name, VFFilter filter) {
        filters[name] = filter;
    }

    VFFilter get_filter(const std::string &name) {
        if (filters.find(name) == filters.end())
            return nullptr;
        return filters[name];
    }

  private:
    std::unordered_map<std::string, VFFilter> filters;

    VFFilterManager() {}
    ~VFFilterManager() {}
};

// register video filter
#define REGISTER_VFFILTER(name, filter)                                        \
    namespace {                                                                \
    struct FilterInitializer_##name {                                          \
        FilterInitializer_##name() {                                           \
            VFFilter name = filter;                                            \
            VFFilterManager::get_instance().register_filter(#name, name);      \
        }                                                                      \
    };                                                                         \
    static FilterInitializer_##name initializer_##name;                        \
    }

VideoFrame bmf_scale_func_with_param(VideoFrame &src_vf, int w, int h,
                                     int mode = 1);
VideoFrame bmf_scale_func(VideoFrame &src_vf, JsonParam param);

VideoFrame bmf_csc_func_with_param(VideoFrame &src_vf,
                                   const hmp::PixelInfo &pixel_info);
VideoFrame bmf_csc_func(VideoFrame &src_vf, JsonParam param);

END_BMF_SDK_NS
