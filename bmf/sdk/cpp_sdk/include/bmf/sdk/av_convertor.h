#include <bmf/sdk/convert_backend.h>
#include <bmf/sdk/ffmpeg_helper.h>

namespace bmf_sdk {

class AVConvertor : public Convertor
{
public:
    AVConvertor(){}
    int media_cvt(VideoFrame &src, const MediaDesc &dp) override {
        try {
            AVFrame *frame = ffmpeg::from_video_frame(src, false);
            src.private_attach<AVFrame>(frame);
        } catch (std::exception &e) {
            BMFLOG(BMF_ERROR) << "convert to AVFrame err: " << e.what();
            return -1;
        }
        return 0;
    }

    int media_cvt_to_videoframe(VideoFrame &src, const MediaDesc &dp) override {
        try {
            const AVFrame *avf = src.private_get<AVFrame>();
            if (!avf) {
                BMFLOG(BMF_ERROR) << "private data is null, please use private_attach before call this api";
                return -1;
            }
            VideoFrame res = ffmpeg::to_video_frame(avf, true);
            src = res;
        } catch (std::exception &e) {
            BMFLOG(BMF_ERROR) << "AVFrame convert to VideoFrame err: " << e.what();
            return -1;
        }
        return 0;
    }
};

static Convertor *av_convert = new AVConvertor();
BMF_REGISTER_CONVERTOR(MediaType::kAVFrame, av_convert);

}
