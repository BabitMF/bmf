#include <bmf/sdk/module_functor.h>
#include <bmf/sdk/video_frame.h>
#include <bmf/sdk/json_param.h>
#include <bmf/sdk/module_registry.h>

using namespace bmf_sdk;

// make sure static module library is linked
BMF_DECLARE_MODULE(VeryfastDenoiseModule)


int main(int argc, char *argv[])
{
    BMF_IMPORT_MODULE(VeryfastDenoiseModule);

    JsonParam json;
    auto denoise = make_sync_func<std::tuple<VideoFrame>, std::tuple<VideoFrame>>(
                         ModuleInfo("VeryfastDenoiseModule", "c++", ""),
                         json);

    static PixelInfo YUV420P = PixelInfo(PixelFormat::PF_YUV420P);
    auto ivf = VideoFrame::make(1920, 1080, YUV420P); 

    auto ovf = denoise(ivf);

    return 0;
}
