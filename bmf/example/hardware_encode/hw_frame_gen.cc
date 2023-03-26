#include <bmf/sdk/bmf.h>
#include <bmf/sdk/packet.h>
#include <bmf/sdk/video_frame.h>
#include <bmf/sdk/ffmpeg_helper.h>
extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavutil/opt.h>
#include <libavutil/frame.h>
};

int64_t nframe = 0;
AVBufferRef* hw_frames_ctx = nullptr;

using namespace bmf_sdk;
class HwFrameGen : public Module
{
public:
    HwFrameGen(int&, bmf_sdk::JsonParam&) {}

    int process(Task &task) {
        int width = 1920, height = 1080;

        auto NV12 = PixelInfo(hmp::PF_NV12, hmp::CS_BT709);
        AVFrame* avfrm = hmp::ffmpeg::hw_avframe_from_device(kCUDA, width, height, NV12, hw_frames_ctx);
        avfrm->pts = nframe++;
        VideoFrame vf = ffmpeg::to_video_frame(avfrm);
        vf.set_pts(avfrm->pts);
        // Add output frame to output queue
        auto output_pkt = Packet(vf);
        output_pkt.set_timestamp(avfrm->pts);
        av_frame_free(&avfrm);

        task.fill_output_packet(0, output_pkt);
        return 0;
    }
};

REGISTER_MODULE_CLASS(HwFrameGen)
