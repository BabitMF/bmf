#include <convert_backend.h>

namespace bmf_sdk {

static std::map<std::string, Convertor*> iConvertors;

BMF_API void set_convertor(std::string &media_type, Convertor convertor)
{
    iConvertors[media_type] = convertor;
}

BMF_API Convertor *get_convertor(std::string &media_type)
{
    if (iConvertors.find(media_type) == iConvertors.end()) {
        BMFLOG(BMF_WARNING) << "the media type is not supported by bmf backend";
        return NULL;
    }
    return iConvertors[media_type];
}

BMF_API VideoFrame *bmf_convert(VideoFrame vf, MediaDesc &dp)
{
    auto convt = get_convertor(dp.media_type);
    convt->format_cvt(dp);
    convt->device_cvt(dp);
    convt->media_cvt(dp);
}

Convertor::Convertor(MediaDesc &dp)
{
}

int Convertor::format_cvt(MediaDesc &dp)
{
}

int Convertor::device_cvt(MediaDesc &dp)
{
}

static Convertor iDefaultBMFConvertor;

BMF_REGISTER_CONVERTOR("bmf", iDefaultBMFConvertor);

} //namespace bmf_sdk
