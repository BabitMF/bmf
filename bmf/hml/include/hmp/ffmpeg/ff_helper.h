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

#include <hmp/imgproc.h>
#include <hmp/format.h>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavutil/pixdesc.h>
#ifdef HMP_ENABLE_CUDA
#include <libavutil/hwcontext_cuda.h>
#include <hmp/cuda/allocator.h>
#endif

} // extern "C"

namespace hmp {
namespace ffmpeg {

static PixelInfo make_pixel_info(const AVCodecParameters &codecpar) {
    return PixelInfo(
        (PixelFormat)codecpar.format,
        ColorModel((ColorSpace)codecpar.color_space,
                   (ColorRange)codecpar.color_range,
                   (ColorPrimaries)codecpar.color_primaries,
                   (ColorTransferCharacteristic)codecpar.color_trc));
}

static PixelInfo make_pixel_info(const AVFrame &avf) {
    auto format = avf.format;
    if (avf.hw_frames_ctx) {
        format = ((AVHWFramesContext *)(avf.hw_frames_ctx->data))->sw_format;
    }

    return PixelInfo((PixelFormat)format,
                     ColorModel((ColorSpace)avf.colorspace,
                                (ColorRange)avf.color_range,
                                (ColorPrimaries)avf.color_primaries,
                                (ColorTransferCharacteristic)avf.color_trc));
}

template <typename T> static PixelFormatDesc make_pixel_desc(const T &v) {
    auto pix_info = make_pixel_info(v);
    return PixelFormatDesc(pix_info.format());
}

static void assign_pixel_info(AVFrame &avf, const PixelInfo &pix_info) {
    avf.format = (AVPixelFormat)pix_info.format();
    avf.colorspace = (AVColorSpace)pix_info.color_model().space();
    avf.color_range = (AVColorRange)pix_info.color_model().range();
    avf.color_primaries = (AVColorPrimaries)pix_info.color_model().primaries();
    avf.color_trc = (AVColorTransferCharacteristic)pix_info.color_model()
                        .transfer_characteristic();
}

static void assign_pixel_info(AVCodecContext &avcc, const PixelInfo &pix_info) {
    avcc.pix_fmt = (AVPixelFormat)pix_info.format();
    avcc.colorspace = (AVColorSpace)pix_info.color_model().space();
    avcc.color_range = (AVColorRange)pix_info.color_model().range();
    avcc.color_primaries = (AVColorPrimaries)pix_info.color_model().primaries();
    avcc.color_trc = (AVColorTransferCharacteristic)pix_info.color_model()
                         .transfer_characteristic();
}

class HMP_API VideoReader {
  public:
    VideoReader(const std::string &fn);

    std::vector<Frame> read(int64_t nframe);

  private:
    class Private;

    std::shared_ptr<Private> self;
};

class HMP_API VideoWriter {
  public:
    VideoWriter(const std::string &fn, int width, int height, int fps,
                const PixelInfo &pix_info, int kbs = 2000);
    ~VideoWriter();

    void write(const std::vector<Frame> &frames);

  private:
    struct Private;

    std::shared_ptr<Private> self;
};

static std::string AVErr2Str_(int rc) {
    char av_error[AV_ERROR_MAX_STRING_SIZE] = {0};
    av_make_error_string(av_error, AV_ERROR_MAX_STRING_SIZE, rc);
    return std::string(av_error);
}

static uint64_t av_hw_frames_ctx_to_stream(const AVBufferRef *hw_frames_ctx_buf,
                                           const char *func = "") {
    if (hw_frames_ctx_buf) {
        auto hw_frames_ctx = (AVHWFramesContext *)hw_frames_ctx_buf->data;
        auto hw_device_ctx = hw_frames_ctx->device_ctx;
        HMP_REQUIRE(hw_device_ctx,
                    "{}: invalid hw_frames_ctx, no device ctx found", func);

#ifdef HMP_ENABLE_CUDA
        if (hw_device_ctx->type == AV_HWDEVICE_TYPE_CUDA) {
            auto cuda_device_ctx = (AVCUDADeviceContext *)hw_device_ctx->hwctx;
            HMP_REQUIRE(cuda_device_ctx,
                        "{}: invalid hwframe, no cuda device ctx found", func);
            return (uint64_t)cuda_device_ctx->stream;
        }
#endif

        HMP_REQUIRE(false, "{}: avframe with device type {} is not supported",
                    func, hw_device_ctx->type);
    }

    return 0;
}

static Device av_hw_frames_ctx_to_device(const AVBufferRef *hw_frames_ctx_buf,
                                         const char *func = "") {
    if (hw_frames_ctx_buf) {
        auto hw_frames_ctx = (AVHWFramesContext *)hw_frames_ctx_buf->data;
        auto hw_device_ctx = hw_frames_ctx->device_ctx;
        HMP_REQUIRE(hw_device_ctx,
                    "{}: invalid hw_frames_ctx, no device ctx found", func);

#ifdef HMP_ENABLE_CUDA
        if (hw_device_ctx->type == AV_HWDEVICE_TYPE_CUDA) {
            auto cuda_device_ctx = (AVCUDADeviceContext *)hw_device_ctx->hwctx;
            HMP_REQUIRE(cuda_device_ctx,
                        "{}: invalid hwframe, no cuda device ctx found", func);
            int index = 0;

            CUcontext tmp;
            cuCtxPushCurrent(cuda_device_ctx->cuda_ctx);
            auto rc = cuCtxGetDevice(&index);
            cuCtxPopCurrent(&tmp);
            HMP_REQUIRE(rc == CUDA_SUCCESS,
                        "{}: get cuda device index failed with rc={}", func,
                        rc);

            return Device(kCUDA, index);
        }
#endif

        HMP_REQUIRE(false, "{}: avframe with device type {} is not supported",
                    func, hw_device_ctx->type);
    }

    return kCPU;
}

static AVBufferRef *av_hw_frames_ctx_from_device(const Device &device,
                                                 int width, int height,
                                                 AVPixelFormat sw_format) {
    if (device.type() == kCPU) {
        return nullptr;
    }

#ifdef HMP_ENABLE_CUDA
    if (device.type() == kCUDA) {
        AVBufferRef *hw_device_ctx = nullptr;
        auto index_str = fmt::format("{}", device.index());
        auto rc =
            av_hwdevice_ctx_create(&hw_device_ctx, AV_HWDEVICE_TYPE_CUDA,
                                   index_str.c_str(), // index=atol(device)
                                   nullptr, AV_CUDA_USE_PRIMARY_CONTEXT);
        HMP_REQUIRE(rc == 0, "create cuda hwdevice failed with rc={}", rc);

        AVBufferRef *hw_frame_ctx = av_hwframe_ctx_alloc(hw_device_ctx);

        av_buffer_unref(&hw_device_ctx);

        AVHWFramesContext *av_hw_frames_ctx =
            (AVHWFramesContext *)hw_frame_ctx->data;
        av_hw_frames_ctx->format = AV_PIX_FMT_CUDA;
        av_hw_frames_ctx->sw_format = sw_format;
        av_hw_frames_ctx->width = width;
        av_hw_frames_ctx->height = height;
        rc = av_hwframe_ctx_init(hw_frame_ctx);
        if (rc < 0) {
            HMP_REQUIRE(false, "av_hwframe_ctx_init failed with rc={}", rc);
        }

        return hw_frame_ctx;
    }

#endif

    HMP_REQUIRE(false, "hwframe with device type {} is not supported",
                device.type());
    return nullptr;
}

static AVFrame *hw_avframe_from_device(const Device &device, int width,
                                       int height, PixelInfo &pix_info,
                                       AVBufferRef *hw_frame_ctx = nullptr,
                                       AVBufferRef *hw_device_ctx = nullptr) {
    AVFrame *avfrm = av_frame_alloc();
    HMP_REQUIRE(avfrm, "to_audio_frame: alloc AVFrame failed");
    avfrm->width = width;
    avfrm->height = height;
    assign_pixel_info(*avfrm, pix_info);
    auto hw_frames_ctx = av_hw_frames_ctx_from_device(
        kCUDA, width, height, (AVPixelFormat)pix_info.format());
    auto ret = av_hwframe_get_buffer(hw_frames_ctx, avfrm, 0);
    HMP_REQUIRE(ret == 0, "get hwframe buffer failed={}", ret);
    avfrm->format = ((AVHWFramesContext *)(hw_frames_ctx->data))->format;
    av_buffer_unref(&hw_frames_ctx);
    return avfrm;
}

static bool is_video_frame(const AVFrame *avf) {
    return avf->width > 0 && avf->height > 0;
}

static std::tuple<bool, ScalarType> from_sample_format(AVSampleFormat format) {
    switch (format) {
    case AV_SAMPLE_FMT_U8:
        return std::make_tuple(false, kUInt8);
    case AV_SAMPLE_FMT_U8P:
        return std::make_tuple(true, kUInt8);
    case AV_SAMPLE_FMT_S16:
        return std::make_tuple(false, kInt16);
    case AV_SAMPLE_FMT_S16P:
        return std::make_tuple(true, kInt16);
    case AV_SAMPLE_FMT_S32:
        return std::make_tuple(false, kInt32);
    case AV_SAMPLE_FMT_S32P:
        return std::make_tuple(true, kInt32);
    case AV_SAMPLE_FMT_FLT:
        return std::make_tuple(false, kFloat32);
    case AV_SAMPLE_FMT_FLTP:
        return std::make_tuple(true, kFloat32);
    case AV_SAMPLE_FMT_DBL:
        return std::make_tuple(false, kFloat64);
    case AV_SAMPLE_FMT_DBLP:
        return std::make_tuple(true, kFloat64);
    default:
        HMP_REQUIRE(false, "unsupported AVSampleFormat {}", format);
    }
}

static AVSampleFormat to_sample_format(ScalarType dtype, bool planer = false) {
    switch (dtype) {
    case kUInt8:
        return planer ? AV_SAMPLE_FMT_U8P : AV_SAMPLE_FMT_U8;
    case kInt16:
        return planer ? AV_SAMPLE_FMT_S16P : AV_SAMPLE_FMT_S16;
    case kInt32:
        return planer ? AV_SAMPLE_FMT_S32P : AV_SAMPLE_FMT_S32;
    case kFloat32:
        return planer ? AV_SAMPLE_FMT_FLTP : AV_SAMPLE_FMT_FLT;
    case kFloat64:
        return planer ? AV_SAMPLE_FMT_DBLP : AV_SAMPLE_FMT_DBL;
    default:
        HMP_REQUIRE(false, "unsupported AVSampleFormat from {}:{}", dtype,
                    planer);
    }
}

namespace {

static void _tensor_info_free(void *info, uint8_t *) {
    dec_ref((TensorInfo *)info);
}

} // namespace

static AVBufferRef *to_av_buffer(const Tensor &d) {
    // FIXME: incorrect size if d is not contiguous
    return av_buffer_create((uint8_t *)d.unsafe_data(), d.nbytes(),
                            _tensor_info_free,             // free
                            inc_ref(d.tensorInfo().get()), // opaque
                            0);                            // flags
}

static TensorList from_audio_frame(const AVFrame *avf) {
    HMP_REQUIRE(avf && avf->extended_data, "from_audio_frame: Invalid AVFrame");
    HMP_REQUIRE(!is_video_frame(avf),
                "from_audio_frame: AVFrame contains no audio data");

    bool planer;
    ScalarType dtype;
    std::tie(planer, dtype) = from_sample_format((AVSampleFormat)avf->format);

    TensorList planes;
    if (planer) {
        int nplane = avf->channels;
        for (int i = 0; i < nplane; ++i) {
            AVFrame *my_avf = av_frame_clone(avf);
            DataPtr data(
                my_avf->extended_data[i],
                [=](void *p) { av_frame_free((AVFrame **)&my_avf); }, kCPU);
            auto plane =
                from_buffer(std::move(data), dtype, {my_avf->nb_samples});
            HMP_REQUIRE(plane.nbytes() <= my_avf->linesize[0],
                        "from_audio_frame: invalid AVFrame");
            planes.push_back(plane);
        }
    } else {
        AVFrame *my_avf = av_frame_clone(avf);
        DataPtr data(
            my_avf->extended_data[0],
            [=](void *p) { av_frame_free((AVFrame **)&my_avf); }, kCPU);
        auto plane = from_buffer(std::move(data), dtype,
                                 {my_avf->nb_samples, my_avf->channels});
        HMP_REQUIRE(plane.nbytes() <= my_avf->linesize[0],
                    "from_audio_frame: invalid AVFrame");
        planes.push_back(plane);
    }

    return planes;
}

static AVFrame *to_audio_frame(const TensorList &planes, const AVFrame *avf_ref,
                               bool use_clone = true) {
    HMP_REQUIRE(avf_ref && !is_video_frame(avf_ref),
                "to_audio_frame: AVFrame contains no audio data");

    // alloc avframe
    AVFrame *avf = use_clone ? av_frame_clone(avf_ref) : av_frame_alloc();
    HMP_REQUIRE(avf, "to_audio_frame: alloc AVFrame failed");
    if (!use_clone) {
        memset(avf, 0, sizeof(*avf));
        avf->format = avf_ref->format;
        avf->channel_layout = avf_ref->channel_layout;
    }
    auto avf_guard = defer([&]() { av_frame_free(&avf); });

    // unref data buffer
    for (int i = 0; i < FF_ARRAY_ELEMS(avf->buf); i++) {
        if (avf->buf[i]) {
            av_buffer_unref(&avf->buf[i]);
        }
        avf->data[i] = nullptr;
    }
    for (int i = 0; i < avf->nb_extended_buf; ++i) {
        if (avf->extended_buf[i]) {
            av_buffer_unref(&avf->extended_buf[i]);
        }
    }
    av_freep(&avf->extended_buf);
    if (avf->extended_data != avf->data) {
        av_free(avf->extended_data);
    }
    avf->extended_data = nullptr;

    // fill data buffer
    ScalarType dtype;
    bool planer;
    std::tie(planer, dtype) = from_sample_format((AVSampleFormat)avf->format);
    HMP_REQUIRE(dtype == planes[0].dtype(),
                "to_audio_frame: dtype is not "
                "match with ref AVFrame, expect "
                "{}, got {}",
                planes[0].dtype(), dtype);

    // Note: channels get from channel layout is not reliable,
    int channels = avf->channels != 0
                       ? avf->channels
                       : av_get_channel_layout_nb_channels(avf->channel_layout);
    int nb_samples = planes[0].size(0);

    if (planer) {
        HMP_REQUIRE(planes[0].dim() == 1, "to_audio_frame: expect plane data "
                                          "is 1D array data for planer audio "
                                          "frame");
        HMP_REQUIRE(
            planes.size() == channels,
            "to_audio_frame: nb_channels not matched, expect {}, got {}",
            channels, planes.size());
    } else {
        HMP_REQUIRE(planes[0].dim() == 2, "to_audio_frame: expect plane data "
                                          "is 2D array data for interleave "
                                          "audio frame");
        HMP_REQUIRE(
            planes.size() == 1 && planes[0].size(1) == channels,
            "to_audio_frame: nb_channels not matched, expect {}, got {}",
            channels, planes[0].size(1));
    }

    if (planes.size() > FF_ARRAY_ELEMS(avf->buf)) {
        auto nb_extended_buf = planes.size() - FF_ARRAY_ELEMS(avf->buf);
        avf->extended_buf = (AVBufferRef **)av_mallocz_array(
            nb_extended_buf, sizeof(*avf->extended_buf));
        avf->extended_data = (uint8_t **)av_mallocz_array(
            planes.size(), sizeof(*avf->extended_data));
        avf->nb_extended_buf = nb_extended_buf;
    } else {
        avf->extended_data = avf->data;
    }

    for (int i = 0; i < planes.size(); ++i) {
        HMP_REQUIRE(planes[i].is_contiguous(),
                    "to_audio_frame: only support contigous data");

        if (i < FF_ARRAY_ELEMS(avf->buf)) {
            avf->buf[i] = to_av_buffer(planes[i]);
            avf->data[i] = avf->buf[i]->data;
            avf->extended_data[i] = avf->data[i];
        } else {
            auto j = i - FF_ARRAY_ELEMS(avf->buf);
            avf->extended_buf[j] = to_av_buffer(planes[i]);
            avf->extended_data[i] = avf->extended_buf[j]->data;
        }
    }
    avf->linesize[0] = planes[0].nbytes();
    avf->channels = channels;
    avf->nb_samples = nb_samples;

    //
    avf_guard.cancel();

    return avf;
}

static AVFrame *to_audio_frame(const TensorList &planes, AVSampleFormat format,
                               uint64_t channel_layout) {
    AVFrame avf_ref;
    memset(&avf_ref, 0, sizeof(avf_ref));
    avf_ref.format = format;
    avf_ref.channel_layout = channel_layout;

    return to_audio_frame(planes, &avf_ref, false);
}

static Tensor get_video_plane(const AVFrame *avf, int plane,
                              const PixelFormatDesc &pix_desc) {
    HMP_REQUIRE(
        avf && plane < FF_ARRAY_ELEMS(avf->data) && avf->data[plane],
        "get_video_plane: Invalid AVFrame or plane index is out of range");
    HMP_REQUIRE(is_video_frame(avf),
                "get_video_plane: AVFrame contains no image data");

    SizeArray shape, strides;
    ScalarType dtype;
    if (pix_desc.defined()) {
        int channels = pix_desc.channels(plane);
        int itemsize = sizeof_scalar_type(pix_desc.dtype());

        dtype = pix_desc.dtype();
        shape = SizeArray{pix_desc.infer_height(avf->height, plane),
                          pix_desc.infer_width(avf->width, plane), channels};
        strides = SizeArray{avf->linesize[plane] / itemsize, channels, 1};
    } else { // unsupported pixel format
        dtype = kUInt8;
        shape = SizeArray{avf->height, // FIXME: invalid height, as we don't
                                       // known exact height
                          abs(avf->linesize[plane]), // fix negative stride
                          1};
        strides = SizeArray{avf->linesize[plane], 1, 1};
    }
    // Note: negative linesize(strides) is possible(ff_vflip), but its
    // representation is different from tensor
    // need port above code if hmp decided to support negative strides

    //
    auto device =
        av_hw_frames_ctx_to_device(avf->hw_frames_ctx, "get_video_plane");

    // don't use av_buffer_ref, as avf->data[plane] != avf->buf[plane]
    AVFrame *myframe = av_frame_clone(avf);
    DataPtr data(
        myframe->data[plane],
        [=](void *p) { av_frame_free((AVFrame **)&myframe); }, device);

    return from_buffer(std::move(data), dtype, shape, strides);
}

/**
 * @brief
 *
 * @param avf
 * @param plane
 * @return Tensor
 */
static Tensor get_video_plane(const AVFrame *avf, int plane) {
    HMP_REQUIRE(
        avf && plane < FF_ARRAY_ELEMS(avf->data) && avf->data[plane],
        "get_video_plane: Invalid AVFrame or plane index is out of range");

    auto pix_desc = make_pixel_desc(*avf);
    return get_video_plane(avf, plane, pix_desc);
}

static int infer_nplanes(AVPixelFormat format) {
    const AVPixFmtDescriptor *desc = av_pix_fmt_desc_get(format);
    int plane_max = 0;
    for (int i = 0; i < desc->nb_components; i++) {
        plane_max = FFMAX(desc->comp[i].plane, plane_max);
    }
    return plane_max + 1;
}

/**
 * @brief Convert AVFrame to Frame
 *
 * @param avf
 * @return Frame
 */
static Frame from_video_frame(const AVFrame *avf) {
    HMP_REQUIRE(avf, "from_video_frame: Invalid AVFrame");

    auto pix_info = make_pixel_info(*avf);
    PixelFormatDesc pix_desc(pix_info.format());

    auto nplanes = infer_nplanes((AVPixelFormat)pix_info.format());
    TensorList planes;
    for (int i = 0; i < nplanes; ++i) {
        HMP_REQUIRE(avf->data[i] && avf->linesize[i],
                    "from_video_frame: Invalid data in AVFrame")
        auto p = get_video_plane(avf, i, pix_desc);
        planes.push_back(p);
    }

    return Frame(planes, avf->width, avf->height, pix_info);
}

// ffmpeg nv12 require
static bool check_frame_memory_is_contiguous(const Frame &frame) {
    if (!frame.storage().defined()) {
        return true;
    }

    char *last_addr = static_cast<char *>(frame.plane(0).unsafe_data());
    int64_t last_nbytes = frame.plane(0).nbytes();

    // check planes' buffer is contiguous
    // auto pix_desc = PixelFormatDesc(frame.format());
    for (int i = 1; i < frame.nplanes(); ++i) {
        char *addr = static_cast<char *>(frame.plane(i).unsafe_data());
        // not continue
        if (addr != last_addr + last_nbytes) {
            return true;
        }
        last_addr = addr;
        last_nbytes = frame.plane(i).nbytes();
    }

    return false;
}

/**
 * @brief convert Frame to AVFrame,
 * if frame.deivce() != kCPU, hw_frames_ctx info must be provided either by
 * avf_ref->hw_frames_ctx or hw_frames_ctx
 * if both avf_ref->hw_frames_ctx and hw_frames_ctx are provided, hw_frames_ctx
 * will be used
 *
 * @param frame
 * @param avf_ref
 * @param hw_frames_ctx
 * @return AVFrame*
 */
static AVFrame *to_video_frame(const Frame &frame,
                               const AVFrame *avf_ref = nullptr,
                               AVBufferRef *hw_frames_ctx = nullptr) {
    HMP_REQUIRE(!frame.pix_desc().defined() ||
                    (frame.pix_desc().dtype() == frame.dtype()),
                "to_video_frame: invalid dtype of Frame");
    HMP_REQUIRE(!avf_ref || is_video_frame(avf_ref),
                "to_video_frame: AVFrame contains no video data");

#ifdef HMP_ENABLE_CUDA
    if (frame.device().type() == kCUDA &&
        (!avf_ref || (avf_ref && !avf_ref->hw_frames_ctx)) &&
        check_frame_memory_is_contiguous(frame)) {
        HMP_REQUIRE(frame.pix_info().format() == hmp::PF_NV12,
                    "cuda hardware encode need NV12 frame");
        auto nv12 = bmf_sdk::PixelInfo(hmp::PF_NV12, hmp::CS_BT709);
        AVFrame *hw_frm = hmp::ffmpeg::hw_avframe_from_device(
            kCUDA, frame.width(), frame.height(), nv12);
        hmp::cuda::d2d_memcpy(hw_frm->data[0], hw_frm->linesize[0],
                              frame.plane(0).data<uint8_t>(), frame.width(),
                              frame.width(), frame.height());
        hmp::cuda::d2d_memcpy(hw_frm->data[1], hw_frm->linesize[1],
                              frame.plane(1).data<uint8_t>(), frame.width(),
                              frame.width(), frame.height() / 2);
        return hw_frm;
    }
#endif

    if (hw_frames_ctx) {
        auto avf_device = av_hw_frames_ctx_to_device(hw_frames_ctx);
        HMP_REQUIRE(avf_device == frame.device(),
                    "to_video_frame: invalid frame on device {}, as "
                    "hw_frame_ctx is specified as {}",
                    frame.device(), avf_device);
    } else {
        hw_frames_ctx = avf_ref ? avf_ref->hw_frames_ctx : nullptr;
        if (!hw_frames_ctx && frame.device().type() == kCUDA) {
            hw_frames_ctx = av_hw_frames_ctx_from_device(
                kCUDA, frame.width(), frame.height(),
                (AVPixelFormat)frame.pix_info().format());
            HMP_INF("created av context for the hardware frame");
        }
        auto avf_device = av_hw_frames_ctx_to_device(hw_frames_ctx);
        if (frame.device().type() != kCPU) {
            HMP_REQUIRE(avf_device == frame.device(),
                        "to_video_frame: invalid frame on device {}, as "
                        "hw_frame_ctx from AVFrame is {}",
                        frame.device(), avf_device);
        }
    }

    // alloc avframe
    AVFrame *avf = avf_ref ? av_frame_clone(avf_ref) : av_frame_alloc();
    HMP_REQUIRE(avf, "to_video_frame: alloc AVFrame failed");
    if (!avf_ref) {
        if (!frame.pix_desc().defined()) {
            // Warning: to avoid magic result, we need avf_ref if pixel format
            // is not supported
            // ref: sws_scale, usePal()
            HMP_WRN("Ref AVFrame is necessary if PixelFormat is not supported "
                    "internally");
        }
        memset(avf, 0, sizeof(*avf));
    }

    // unref data buffer
    for (int i = 0; i < FF_ARRAY_ELEMS(avf->buf); i++) {
        if (avf->buf[i]) {
            av_buffer_unref(&avf->buf[i]);
        }
        // don't zero it as it may contains some magic data, ref: av_frame_clone
        // ref: sws_scale, usePal()
        // avf->data[i] = nullptr;
    }

    // fill data buffer
    avf->width = frame.width();
    avf->height = frame.height();
    assign_pixel_info(*avf, frame.pix_info());
    if (hw_frames_ctx) {
        avf->format = ((AVHWFramesContext *)(hw_frames_ctx->data))->format;
    }
    for (int i = 0; i < frame.nplanes(); ++i) {
        auto plane = frame.plane(i).view(
            {frame.plane(i).size(0),
             -1}); // ensure shape is (height, width*channels)
        if (plane.stride(1) != 1) {
            av_frame_free(&avf);
            HMP_REQUIRE(false, "to_video_frame: plane data is not contiguous");
        }

        avf->buf[i] = to_av_buffer(plane);
        avf->data[i] = avf->buf[i]->data;
        avf->linesize[i] = plane.stride(0) * plane.itemsize(); // in bytes
    }
    //
    if (!avf->extended_data) {
        avf->extended_data = avf->data;
    } else {
        HMP_REQUIRE(
            avf->extended_data == avf->data,
            "to_video_frame: invalid extended_data, need equal to data");
    }

    //
    if (hw_frames_ctx) {
        if (avf->hw_frames_ctx) { // clone from avf_ref
            av_buffer_unref(&avf->hw_frames_ctx);
        }
        avf->hw_frames_ctx = av_buffer_ref(hw_frames_ctx);
    }

    if (frame.device().type() == kCPU && avf->hw_frames_ctx) { // Fix mismatch
        av_buffer_unref(&avf->hw_frames_ctx);
        avf->format = (AVPixelFormat)frame.format();
    }

    return avf;
}
} // namespace ffmpeg
} // namespace hmp
