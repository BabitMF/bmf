#pragma once

#include <bmf/sdk/common.h>
#include <bmf/sdk/video_frame.h>
#include <bmf/sdk/media_description.h>

namespace bmf_sdk {

class BMF_API Convertor
{
public:
    Convertor();
    virtual ~Convertor() {}
    virtual int media_cvt(VideoFrame &src, const MediaDesc &dp);
    virtual int media_cvt_to_videoframe(VideoFrame &src, const MediaDesc &dp);
    virtual VideoFrame format_cvt(VideoFrame &src, const MediaDesc &dp);
    virtual VideoFrame device_cvt(VideoFrame &src, const MediaDesc &dp);
};


BMF_API void set_convertor(const MediaType &media_type, Convertor* convertor);
BMF_API Convertor *get_convertor(const MediaType &media_type);

BMF_API VideoFrame bmf_convert(VideoFrame& src_vf, const MediaDesc &src_dp, const MediaDesc &dst_dp);

template<typename ...Args>
struct Register {
    using RegisterFunc = void(*)(Args ...args);
    explicit Register(RegisterFunc func, Args...args){
        func(args...);
    }
};

#define BMF_REGISTER_CONVERTOR(media_type, convertor) \
        static Register<const MediaType&, Convertor*> \
            __i##media##convertor(set_convertor, media_type, convertor);

} //namespace bmf_sdk
