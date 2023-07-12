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

//MediaParam
template <typename T>
class MediaParam : public std::optional<T> {
public:
    MediaParam(MediaDesc* m = nullptr) : media(m) {}

    MediaParam(const MediaParam& mp) : std::optional<T>(mp) {}

    MediaParam(MediaParam&& mp) : std::optional<T>(mp) {}

    MediaParam& operator=(const MediaParam& mp) {
        std::optional<T>::operator=(mp);
        return *this;
    }

    MediaParam& operator=(T val) {
        std::optional<T>::emplace(val);
    }

    MediaDesc& operator()(T val) {
        std::optional<T>::emplace(val);
        return *media;
    }

    const T &operator()() const {
        return std::optional<T>::value();
    }

private:
    //pinned after initialized
    MediaDesc* const media;
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

