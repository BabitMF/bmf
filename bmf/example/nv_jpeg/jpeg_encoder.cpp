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

#include <bmf/sdk/bmf_av_packet.h>
#include <bmf/sdk/common.h>
#include <bmf/sdk/module.h>
#include <bmf/sdk/module_registry.h>
#include <bmf/sdk/task.h>
#include <bmf/sdk/video_frame.h>
#include <bmf/sdk/ffmpeg_helper.h>
#include <bmf/sdk/error_define.h>

extern "C" {
#include <libavcodec/packet.h>
}

#include <nvjpeg.h>

#include <cstdlib>
#include <fstream>
#include <mutex>
#include <omp.h>
#include <queue>
#include <unordered_map>

USE_BMF_SDK_NS

#define CHECK_NVJPEG(func, errer_log, _ret, WHETHER_RET)   \
    do {                                                   \
        auto status = func;                                \
        if (status != NVJPEG_STATUS_SUCCESS) {             \
            BMFLOG_NODE(BMF_ERROR, node_id_) << errer_log; \
            if (WHETHER_RET)                               \
                return _ret;                               \
        }                                                  \
    } while (0);

class jpeg_encoder : public bmf_sdk::Module {
public:
    jpeg_encoder(int node_id, bmf_sdk::JsonParam option) : bmf_sdk::Module(node_id, option) {
        if (option.has_key("width") && option.json_value_["width"].is_number()) {
            m_encode_w = option.json_value_["width"];
        }
        if (option.has_key("height") && option.json_value_["height"].is_number()) {
            m_encode_h = option.json_value_["height"];
        }
        BMFLOG_NODE(BMF_INFO, node_id_) << "encode frame size is: "
                                        << m_encode_w << "x" << m_encode_h;
        if (option.has_key("quality") && option.json_value_["quality"].is_number()) {
            m_encode_q = option.json_value_["quality"];
        }
        BMFLOG_NODE(BMF_INFO, node_id_) << "encode param set to " << m_encode_q;
    }

    ~jpeg_encoder() { close(); }

    virtual int init() {
        // cudaStreamCreate(&m_stream);
        m_stream = nullptr;

        // nvjpeg
        CHECK_NVJPEG(nvjpegCreateSimple(&m_handle),
                     "nvjpeg create handle error. ", -1, true);
        CHECK_NVJPEG(nvjpegEncoderParamsCreate(m_handle, &m_enc_param, m_stream),
                     "nvjpeg create enc param error. ", -1, true);
        CHECK_NVJPEG(nvjpegEncoderStateCreate(m_handle, &m_enc_state, m_stream),
                     "nvjpeg create jpeg enc state error. ", -1, true);

        // set param
        CHECK_NVJPEG(nvjpegEncoderParamsSetQuality(m_enc_param, m_encode_q, m_stream),
                     "nvjpeg set enc quality error. ", -1, true);
        auto status = nvjpegEncoderParamsSetEncoding(m_enc_param,
                                                     NVJPEG_ENCODING_BASELINE_DCT,
                                                     m_stream);
        status = nvjpegEncoderParamsSetOptimizedHuffman(m_enc_param, 2, m_stream);
        status = nvjpegEncoderParamsSetSamplingFactors(m_enc_param, NVJPEG_CSS_420, m_stream);

        return 0;
    }

    virtual int close() {
        if (m_enc_param) {
            nvjpegEncoderParamsDestroy(m_enc_param);
            m_enc_param = nullptr;
        }
        if (m_enc_state) {
            nvjpegEncoderStateDestroy(m_enc_state);
            m_enc_state = nullptr;
        }
        if (m_handle) {
            CHECK_NVJPEG(nvjpegDestroy(m_handle),
                         "nvjpeg handle destroy error. ", -1, true);
            m_handle = nullptr;
        }
        if (m_stream) {
            cudaStreamDestroy(m_stream);
            m_stream = nullptr;
        }
        BMFLOG_NODE(BMF_INFO, node_id_) << "encode " << m_frame_count << " frames.";
        return 0;
    }

    virtual int process(bmf_sdk::Task &task) {
        auto input = task.get_inputs();
        int label = input.begin()->first;
        for (auto it = input.begin(); it != input.end(); it++) {
            label = it->first;
            bmf_sdk::Packet pkt;
            while (task.pop_packet_from_input_queue(label, pkt)) {
                if (pkt.timestamp() == bmf_sdk::Timestamp::BMF_EOF) {
                    task.fill_output_packet(label, bmf_sdk::Packet::generate_eof_packet());
                    task.set_timestamp(bmf_sdk::Timestamp::DONE);
                    return 0;
                }
                auto start = std::chrono::steady_clock::now();

                auto vf = pkt.get<VideoFrame>();
                auto width = vf.width();
                auto height = vf.height();
                // FIX auto planes = frame.get_planes();

                auto format = vf.frame().format();
                // start encoding
                nvjpegImage_t image = {0};
                if (vf.device() == kCUDA) {
                    if (2 != vf.frame().nplanes()) {
                        BMFLOG_NODE(BMF_INFO, node_id_) << "input video frame error. must be nv12 format has 2 planes. this has "
                                                        << vf.frame().nplanes() << " planes.";
                    }

                    if (!vf_rgb)
                        vf_rgb = VideoFrame::make(width, height, 3, kNCHW, kCUDA);
                    Tensor t_img = vf_rgb.image().data();
                    hmp::img::yuv_to_rgb(t_img, vf.frame().data(), NV12);
                    if (!vf_yuv_from_rgb)
                        vf_yuv_from_rgb = VideoFrame::make(width, height, H420, kCUDA);
                    TensorList tl = vf_yuv_from_rgb.frame().data();
                    hmp::img::rgb_to_yuv(tl, vf_rgb.image().data(), H420);

                    vf_final_p = &vf_yuv_from_rgb;
                } else {
                    BMFLOG_NODE(BMF_INFO, node_id_) << "frame comes from cpu decoder.";
                    if (format != hmp::PF_YUV420P) {
                        vf_rfmt = ffmpeg::reformat(vf, "yuv420p");
                        BMFLOG_NODE(BMF_INFO, node_id_) << "encode: frame pixel format to yuv420p.";
                        vf_final_p = &vf_rfmt;
                    } else {
                        vf_final_p = &vf;
                    }

                }

                if (m_encode_w && m_encode_h) {
                    if (!vf_resize)
                        vf_resize = VideoFrame::make(m_encode_w, m_encode_h, H420, kCUDA);
                    TensorList tl_resize = vf_resize.frame().data();
                    vf_cuda = vf_final_p->cuda();
                    hmp::img::yuv_resize(tl_resize, vf_cuda.frame().data(), H420);

                    vf_final_p = &vf_resize;
                    width = m_encode_w;
                    height = m_encode_h;
                }
                image.channel[0] = vf_final_p->frame().plane(0).data<uint8_t>();
                image.channel[1] = vf_final_p->frame().plane(1).data<uint8_t>();
                image.channel[2] = vf_final_p->frame().plane(2).data<uint8_t>();
                image.pitch[0] = vf_final_p->frame().plane(0).stride(0);
                image.pitch[1] = vf_final_p->frame().plane(1).stride(0);
                image.pitch[2] = vf_final_p->frame().plane(2).stride(0);

                size_t length = 0;
                CHECK_NVJPEG(nvjpegEncodeYUV(m_handle, m_enc_state, m_enc_param,
                                             &image, NVJPEG_CSS_420, width, height, m_stream),
                             "nvjpeg encode error. ", -1, true);
                CHECK_NVJPEG(nvjpegEncodeRetrieveBitstream(m_handle, m_enc_state, nullptr, &length, m_stream),
                             "nvjpeg get bit stream data error. ", -1, true);
                cudaStreamSynchronize(m_stream);

                // init output
                auto bmf_avpkt = BMFAVPacket(length);
                auto data = bmf_avpkt.data_ptr();
                CHECK_NVJPEG(nvjpegEncodeRetrieveBitstream(m_handle, m_enc_state, (uint8_t *) data, &length, m_stream),
                             "nvjpeg get bit stream data error. ", -1, true);
                cudaStreamSynchronize(m_stream);

                auto packet = Packet(bmf_avpkt);
                packet.set_timestamp(vf_final_p->pts());

                task.fill_output_packet(label, packet);

                auto end = std::chrono::steady_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
                BMFLOG_NODE(BMF_INFO, node_id_)  << "encode jpeg " << width << "x"
                                                 << height << " to " << length << " bytes. using "
                                                 << ((float) duration) * 0.001f << " ms";
                m_frame_count++;
            }
        }
        return 0;
    }

private:
    nvjpegHandle_t m_handle = nullptr;
    nvjpegEncoderParams_t m_enc_param = nullptr;
    nvjpegEncoderState_t m_enc_state = nullptr;

    cudaStream_t m_stream = nullptr;

    VideoFrame vf_rgb;
    VideoFrame vf_yuv_from_rgb;
    VideoFrame vf_resize;
    VideoFrame vf_rfmt;
    VideoFrame vf_cuda;
    VideoFrame* vf_final_p;
    int32_t m_encode_w = 0;
    int32_t m_encode_h = 0;
    int32_t m_encode_q = 99;
    int32_t m_frame_count = 0;

    PixelInfo NV12 = PixelInfo(hmp::PF_NV12, hmp::CS_BT470BG);
    PixelInfo H420 = PixelInfo(hmp::PF_YUV420P, hmp::CS_BT709);
};

REGISTER_MODULE_CLASS(jpeg_encoder)
