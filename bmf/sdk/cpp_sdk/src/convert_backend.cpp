#include <bmf/sdk/convert_backend.h>
#include <bmf/sdk/sdk_interface.h>
#include <bmf/sdk/log.h>

#include <unordered_map>

namespace bmf_sdk {


static std::unordered_map<MediaType, Convertor*> iConvertors;

BMF_API void set_convertor(const MediaType &media_type, Convertor* convertor)
{
    iConvertors[media_type] = convertor;
}

BMF_API Convertor *get_convertor(const MediaType &media_type)
{
    if (iConvertors.find(media_type) == iConvertors.end()) {
        BMFLOG(BMF_WARNING) << "the media type is not supported by bmf backend";
        return NULL;
    }
    return iConvertors[media_type];
}

BMF_API VideoFrame bmf_convert(VideoFrame vf, MediaDesc &dp)
{
    auto convt = get_convertor(MediaType::kBMFVideoFrame);
    if (dp.media_type.has_value()) {
        convt = get_convertor(dp.media_type());
    }

    auto format_vf = convt->format_cvt(vf, dp);
    auto device_vf = convt->device_cvt(format_vf, dp);
    convt->media_cvt(device_vf, dp);
    return device_vf;
}

Convertor::Convertor()
{

}

VideoFrame Convertor::format_cvt(VideoFrame &src, MediaDesc &dp)
{
    // do scale
    //if (dp.width.has_value() || dp.height.has_value()) {
    //    f.push_back(scale_filter(""));
    //}

    //for 

    // do 
}

VideoFrame Convertor::device_cvt(VideoFrame &src, MediaDesc &dp)
{
}

int Convertor::media_cvt(VideoFrame &src, MediaDesc &dp)
{
}

static Convertor* iDefaultBMFConvertor = new Convertor();

BMF_REGISTER_CONVERTOR(MediaType::kBMFVideoFrame, iDefaultBMFConvertor);

} //namespace bmf_sdk
