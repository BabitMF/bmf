
#include <bmf/sdk/ffmpeg_helper.h>
#include <bmf/sdk/log_buffer.h>

namespace bmf_sdk{
namespace {

class FFMPEGRegistry
{
public:
    FFMPEGRegistry()
    {
        LogBuffer::register_av_log_set_callback((void*)av_log_set_callback);
    }
};

FFMPEGRegistry sFFRegistery;

}}; //bmf_sdk