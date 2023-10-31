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

#include <bmf/sdk/common.h>
#include <hmp/imgproc/formats.h>
#include <hmp/core/device.h>
#include <bmf/sdk/video_frame.h>
#include <optional>

USE_BMF_SDK_NS

BEGIN_BMF_SDK_NS

using MediaType = OpaqueDataKey::Key;

class MediaDesc {

    // MediaParam
    template <typename T> class MediaParam : public std::optional<T> {
      public:
        MediaParam(MediaDesc *m = nullptr) : media(m) {}
        MediaParam(const MediaParam &mp) : std::optional<T>(mp) {}

        MediaParam(MediaParam &&mp) : std::optional<T>(mp) {}

        MediaParam &operator=(const MediaParam &mp) {
            std::optional<T>::operator=(mp);
            return *this;
        }

        MediaParam &operator=(T val) { std::optional<T>::emplace(val); }

        MediaDesc &operator()(T val) {
            std::optional<T>::emplace(val);
            return *media;
        }

        const T &operator()() const { return std::optional<T>::value(); }

      private:
        // pinned after initialized
        MediaDesc *const media = nullptr;
    };

  public:
    MediaParam<int> width{this};
    MediaParam<int> height{this};
    MediaParam<hmp::PixelFormat> pixel_format{this};
    MediaParam<hmp::ColorSpace> color_space{this};
    MediaParam<hmp::Device> device{this};
    MediaParam<MediaType> media_type{this};
};

END_BMF_SDK_NS
