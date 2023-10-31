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

// Header only ffmpeg helper functions
// Note: Any source include this header may add ffmpeg dependecies

#include <bmf/sdk/video_frame.h>
#include <bmf/sdk/audio_frame.h>
#include <bmf/sdk/bmf_av_packet.h>
#include <bmf/sdk/simple_filter_graph.h>
#include <bmf/sdk/error_define.h>
#include <bmf/sdk/exception_factory.h>
#include <hmp/ffmpeg/ff_helper.h>
extern "C" {
#include <libswscale/swscale.h>
#include <libavutil/pixdesc.h>
}

namespace bmf_sdk {

template <> struct OpaqueDataInfo<AVFrame> {
    const static int key = OpaqueDataKey::kAVFrame;

    static OpaqueData construct(const AVFrame *avframe) {
        auto avf = av_frame_clone(avframe);
        return OpaqueData(avf, [](void *ptr) {
            auto avf = (AVFrame *)ptr;
            av_frame_free(&avf);
        });
    }
};

template <> struct OpaqueDataInfo<AVPacket> {
    const static int key = OpaqueDataKey::kAVPacket;

    static OpaqueData construct(const AVPacket *avpacket) {
        auto avp = av_packet_clone(avpacket);
        return OpaqueData(avp, [](void *ptr) {
            auto avp = (AVPacket *)ptr;
            av_packet_free(&avp);
        });
    }
};

namespace ffmpeg {

/**
 * @brief Convert VideoFrame to AVFrame, if AVFrame have been attach to this
 * VideoFrame,
 * it will be used as a prototype
 *
 * @param vf
 * @param force_private if true, AVFrame must exists in VideoFrame as private
 * data
 * @return AVFrame*
 */
static AVFrame *from_video_frame(const VideoFrame &vf,
                                 bool force_private = true) {
    auto avf_ref = vf.private_get<AVFrame>();
    if (!avf_ref && force_private) {
        throw std::runtime_error(
            "No AVFrame found in VideoFrame as private_data");
    }

    auto avf = hmp::ffmpeg::to_video_frame(vf.frame(), avf_ref);
    avf->pts = vf.pts();

    // FIXME: remove after encoder&decoder update
    std::string s_tb = std::to_string(vf.time_base().num) + "," +
                       std::to_string(vf.time_base().den);
    av_dict_set(&avf->metadata, "time_base", s_tb.c_str(), 0);

    if (avf->hw_frames_ctx) {
        // FIXME: the caller may need to sync stream between vf and avf,
        // we can't do that in this func, as vf does not actually own
        // vf->stream()
    }

    return avf;
}

/**
 * @brief Convert AVFrame to VideoFrame, VideoFrame share the same buffer with
 * AVFrame
 *
 * @param avf
 * @param attach if true, avf will attach to VideoFrame as private data
 * @return VideoFrame
 */
static VideoFrame to_video_frame(const AVFrame *avf, bool attach = true) {
    auto vf = VideoFrame(hmp::ffmpeg::from_video_frame(avf));
    if (attach) {
        vf.private_attach<AVFrame>(avf);
    }
    vf.set_pts(avf->pts);

    //
    vf.set_stream(hmp::ffmpeg::av_hw_frames_ctx_to_stream(avf->hw_frames_ctx,
                                                          "to_video_frame"));

    return vf;
}

static AVFrame *from_audio_frame(const AudioFrame &af,
                                 bool force_private = true) {
    auto aaf_ref = af.private_get<AVFrame>();
    if (!aaf_ref && force_private) {
        throw std::runtime_error(
            "No AVFrame found in AudioFrame as private_data");
    }

    AVFrame *aaf;
    if (aaf_ref) {
        aaf = hmp::ffmpeg::to_audio_frame(af.planes(), aaf_ref);
    } else {
        auto format = hmp::ffmpeg::to_sample_format(af.dtype(), af.planer());
        aaf = hmp::ffmpeg::to_audio_frame(af.planes(), format, af.layout());
    }
    aaf->pts = af.pts();
    aaf->sample_rate = af.sample_rate();

    // FIXME: remove after encoder&decoder update
    std::string s_tb = std::to_string(af.time_base().num) + "," +
                       std::to_string(af.time_base().den);
    av_dict_set(&aaf->metadata, "time_base", s_tb.c_str(), 0);

    return aaf;
}

static AudioFrame to_audio_frame(const AVFrame *aaf, bool attach = true) {
    auto data = hmp::ffmpeg::from_audio_frame(aaf);
    auto planer = av_sample_fmt_is_planar((AVSampleFormat)aaf->format);
    auto af = AudioFrame::make(data, aaf->channel_layout, planer);
    if (attach) {
        af.private_attach(aaf);
    }
    af.set_pts(aaf->pts);
    af.set_sample_rate(aaf->sample_rate);

    return af;
}

/**
 * @brief Convert BMFAVPacket to AVPacket, if force_private is true, AVPacket
 * must
 * be attach to pkt as private data
 *
 * @param pkt
 * @param force_private
 * @return AVPacket*
 */
static AVPacket *from_bmf_av_packet(const BMFAVPacket &pkt,
                                    bool force_private = true) {
    auto avp_ref = pkt.private_get<AVPacket>();
    if (!avp_ref && force_private) {
        throw std::runtime_error(
            "No AVPacket found in BMFAVPacket as private_data");
    }

    auto avp = avp_ref ? av_packet_clone(avp_ref) : av_packet_alloc();
    if (!avp_ref) {
        memset(avp, 0, sizeof(*avp));
    }

    if (avp->buf) {
        av_buffer_unref(&avp->buf);
        avp->data = nullptr;
    }

    try {
        auto data = pkt.data();
        avp->buf = hmp::ffmpeg::to_av_buffer(data);
        avp->data = avp->buf->data;
        avp->size = avp->buf->size;
        avp->pts = pkt.pts();
    } catch (...) {
        av_packet_free(&avp);
        throw;
    }

    return avp;
}

/**
 * @brief convert AVPacket to BMFAVPacket
 *
 * @param avp_
 * @param attach
 * @return BMFAVPacket
 */
static BMFAVPacket to_bmf_av_packet(const AVPacket *avp_, bool attach = true) {
    //
    auto avp = av_packet_clone(avp_);
    auto data_ptr = DataPtr(
        avp->data, [=](void *) { av_packet_free((AVPacket **)&avp); }, kCPU);
    auto data = from_buffer(std::move(data_ptr), kUInt8, {int64_t(avp->size)});

    //
    auto pkt = BMFAVPacket(data);
    if (attach) {
        pkt.private_attach(avp);
    }
    pkt.set_pts(avp->pts);

    return pkt;
}

struct AVFrameDeleter {
    void operator()(AVFrame *avf) { av_frame_free(&avf); }
};
using AVFramePtr = std::unique_ptr<AVFrame, AVFrameDeleter>;

static AVFramePtr make_avframe(int width, int height, int format,
                               int align = 32) {
    AVFramePtr avf(av_frame_alloc());
    avf->width = width;
    avf->height = height;
    avf->format = format;
    auto res = av_frame_get_buffer(avf.get(), align);
    HMP_REQUIRE(res == 0, "make_avframe: allocate buffer failed");

    return std::move(avf);
}

static SimpleFilterGraph init_reformat_filter(AVFrame *av_frame,
                                              const std::string &format,
                                              std::string flags) {
    SimpleFilterGraph simple_filter_graph;
    std::string filter_desc = flags.empty() ? "" : ("sws_flags=" + flags + ";");
    filter_desc += "[i0_0]format=pix_fmts=" + format + "[o0_0]";
    simple_filter_graph.init(av_frame, filter_desc);
    return simple_filter_graph;
}

static VideoFrame reformat(const VideoFrame &vf, const std::string &format_str,
                           SimpleFilterGraph filter_graph = SimpleFilterGraph(),
                           std::string flags = "") {
    HMP_REQUIRE(vf.device().type() == kCPU,
                "ffmpeg::reformat only support CPU data")
    std::vector<AVFrame *> result_frames;

    AVFrame *av_frame = from_video_frame(vf, false);

    if (filter_graph.filter_graph_ == nullptr) {
        filter_graph = init_reformat_filter(av_frame, format_str, flags);
    }

    // allocate dest AVFrame
    filter_graph.get_filter_frame(av_frame, result_frames);
    if (result_frames.size() != 1) {
        BMF_Error(BMF_TranscodeError, "filter process error");
    }

    auto dst_vf = to_video_frame(result_frames[0], false);

    av_frame_free(&result_frames[0]);
    av_frame_free(&av_frame);

    dst_vf.copy_props(vf);
    return dst_vf;
}

static VideoFrame reformat(const VideoFrame &vf, AVPixelFormat format,
                           SimpleFilterGraph filter_graph = SimpleFilterGraph(),
                           std::string flags = "") {
    return reformat(vf, av_get_pix_fmt_name(format), filter_graph, flags);
}

/*
static VideoFrame reformat(const VideoFrame &vf, AVPixelFormat format, void
**context = nullptr)
{
    HMP_REQUIRE(vf.device().type() == kCPU, "ffmpeg::reformat only support CPU
data")
    auto cached_context = context ? *((struct SwsContext**)context) : nullptr;
    struct SwsContext *sws_context = nullptr;
    AVFramePtr dst_frame;

    if(vf.is_image()){
        HMP_REQUIRE(vf.dtype() == kUInt8 || vf.dtype() == kUInt16,
                 "ffmpeg::reformat only support Image with dtype kUInt8 or
kUInt16")

        // infer source AVPixFormat
        int avformat = hmp::PF_NONE;
        switch(vf.image().nchannels()){
            case 1:
                avformat = vf.dtype() == kUInt8 ? hmp::PF_GRAY8 :
hmp::PF_GRAY16;
                break;
            case 3:
                avformat = vf.dtype() == kUInt8 ? hmp::PF_RGB24 : hmp::PF_RGB48;
                break;
            case 4:
                avformat = vf.dtype() == kUInt8 ? hmp::PF_RGBA32 :
hmp::PF_RGBA64;
                break;
            default:
                HMP_REQUIRE(false, "ffmpeg::reformat unsupported image chnanels
{}", vf.image().nchannels());
        }

        // change to HWC layout
        auto tmp = vf.image().format() == kNCHW ? vf.image().to(kNHWC, true) :
vf.image();

        // allocate dest AVFrame
        dst_frame = make_avframe(vf.width(), vf.height(), format);

        sws_context = sws_getCachedContext(cached_context, vf.width(),
vf.height(), (AVPixelFormat)avformat,
                                                dst_frame->width,
dst_frame->height, format,
                                                SWS_BILINEAR, NULL, NULL, NULL);

        uint8_t *src_data[4] = {(uint8_t*)tmp.data().unsafe_data(), 0, 0, 0};
        int src_linesize[4] = {int(tmp.data().stride(0) *
tmp.data().itemsize()), 0, 0, 0};
        int height = sws_scale(sws_context, src_data, src_linesize, 0,
vf.height(),
                              dst_frame->data, dst_frame->linesize);
        HMP_REQUIRE(height == vf.height(), "ffmpeg::reformat internal error");
    }
    else {
        auto src_frame = AVFramePtr(from_video_frame(vf, false));
        dst_frame = make_avframe(vf.width(), vf.height(), format);

        sws_context = sws_getCachedContext(cached_context, vf.width(),
vf.height(), (AVPixelFormat)src_frame->format,
                                                dst_frame->width,
dst_frame->height, format,
                                                SWS_BILINEAR, NULL, NULL, NULL);
        int height = sws_scale(sws_context, src_frame->data,
src_frame->linesize, 0, vf.height(),
                               dst_frame->data, dst_frame->linesize);
        HMP_REQUIRE(height == vf.height(), "ffmpeg::reformat internal error");
    }

    if(context){
        *context = sws_context; //managed by caller
    }
    else{
        sws_freeContext(sws_context);
    }

    auto dst_vf = to_video_frame(dst_frame.get(), false);
    dst_vf.copy_props(vf);
    return dst_vf;
}


static VideoFrame reformat(const VideoFrame &vf, const std::string &format_str,
void **context = nullptr)
{
    const AVPixelFormat format = av_get_pix_fmt(format_str.c_str());
    return reformat(vf, format, context);
}
*/

} // namespace ffmpeg
} // namespace bmf_sdk
