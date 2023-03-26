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

#include <map>
#include <iostream>

#include <ffmpeg/ffmpeg_utils.h>

namespace hmp{
namespace ffmpeg{



struct VideoReader::Private
{
    std::shared_ptr<AVFormatContext> avfc;
    std::shared_ptr<AVCodecContext>  avcc;
    std::shared_ptr<AVPacket> avpkt;
    std::shared_ptr<AVFrame> avframe;

    //
    PixelInfo pix_info;
    int streamIndex;
};


VideoReader::VideoReader(const std::string &fn)
{
    self = std::make_shared<Private>();

    //
    AVFormatContext *avfc = avformat_alloc_context();
    HMP_REQUIRE(avfc, "FFMPEG: allocate AVFormatContext failed");
    self->avfc = decltype(self->avfc)(avfc, std::ptr_fun(avformat_free_context));

    auto rc = avformat_open_input(&avfc, fn.c_str(), NULL, NULL);
    HMP_REQUIRE(rc == 0, "FFMPEG: open file {} failed", fn);

    rc = avformat_find_stream_info(avfc, NULL);
    HMP_REQUIRE(rc == 0, "FFMPEG: failed to get stream info");

    //using first video stream
    AVStream *avs = 0;
    for(int i = 0; i < avfc->nb_streams; i ++){
        if(avfc->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO){
            avs = avfc->streams[i];
            self->streamIndex = i;
            break;
        }
    }
    HMP_REQUIRE(avs != nullptr, "FFMPEG: no video stream found");
    self->pix_info = make_pixel_info(*avs->codecpar);

    auto avc = avcodec_find_decoder(avs->codecpar->codec_id);
    HMP_REQUIRE(avc,
     "FFMPEG: failed to find the codec with code_id {}", avs->codecpar->codec_id);

    auto avcc = avcodec_alloc_context3(avc);
    HMP_REQUIRE(avcc, "FFMPEG: failed to alloc AVCodecContext");
    self->avcc = std::shared_ptr<AVCodecContext>(avcc,
         [](AVCodecContext *ptr){ avcodec_free_context(&ptr); });

    rc = avcodec_parameters_to_context(avcc, avs->codecpar);
    HMP_REQUIRE(rc == 0, "FFMPEG: failed to fill codec context");

    rc = avcodec_open2(avcc, avc, NULL);
    HMP_REQUIRE(rc == 0, "FFMPEG: failed to open codec");

    self->avpkt = std::shared_ptr<AVPacket>(av_packet_alloc(), 
        [](AVPacket *pkt) { av_packet_free(&pkt); });

    self->avframe = std::shared_ptr<AVFrame>(av_frame_alloc(), 
        [](AVFrame *frame) { av_frame_free(&frame); });

    HMP_REQUIRE(self->avpkt && self->avframe,
         "FFMPEG: alloc AVPacket or AVFrame failed");
}

std::vector<Frame> VideoReader::read(int64_t nframe)
{
    auto &pix_info = self->pix_info;
    std::vector<Frame> frames;

    int nread = 0;
    while(nread < nframe){
        auto rc = av_read_frame(self->avfc.get(), self->avpkt.get()); 
        if(rc >= 0){
            if(self->avpkt->stream_index == self->streamIndex){
                rc = avcodec_send_packet(self->avcc.get(), self->avpkt.get());
                HMP_REQUIRE(rc >= 0, "FFMPEG: decode failed, rc={}", rc);

                while(rc >= 0){
                    rc = avcodec_receive_frame(self->avcc.get(), self->avframe.get());
                    if(rc == AVERROR(EAGAIN) || rc == AVERROR_EOF){
                        break;
                    }
                    else{
                        HMP_REQUIRE(rc >= 0 , "FFMPEG: receive frame failed");
                    }

                    auto f = from_video_frame(self->avframe.get());
                    frames.push_back(f);

                    av_frame_unref(self->avframe.get());
                    nread += 1;
                }
            }

            av_packet_unref(self->avpkt.get());
        }
        else{
            if(rc != AVERROR_EOF){
                HMP_WRN("ReadFrame failed with error {}", AVErr2Str(rc));
            }
            break;
        }
    }

    return frames;
}


struct VideoWriter::Private
{
    std::shared_ptr<AVFormatContext> avfc;
    std::shared_ptr<AVCodecContext>  avcc;
    std::shared_ptr<AVPacket> avpkt;
    AVStream *avs;

    //
    PixelInfo pix_info;
    int streamIndex;

    //
    int64_t pts;
    int64_t fps;
    int64_t ptsStep; //[FIXME] workaround to make it works
};

//https://libav.org/documentation/doxygen/master/encode__video_8c_source.html
static void encodeVideo(AVFormatContext *avfc, AVCodecContext *avcc, AVFrame *avframe, AVPacket *avpkt)
{
    auto rc = avcodec_send_frame(avcc, avframe);

    while (rc >= 0){
        rc = avcodec_receive_packet(avcc, avpkt);
        if (rc == AVERROR(EAGAIN) || rc == AVERROR_EOF){
            break;
        }
        HMP_REQUIRE(rc >= 0,
                    "VideoWriter: receiving packet failed with error {}", AVErr2Str(rc));

        rc = av_interleaved_write_frame(avfc, avpkt);
        av_packet_unref(avpkt);
        HMP_REQUIRE(rc == 0,
                    "VideoWriter: write packet failed with error {}", AVErr2Str(rc));
    }
}


VideoWriter::VideoWriter(const std::string &fn, int width, int height, int fps, const PixelInfo &pix_info, int kbs)
{
    self = std::make_shared<Private>();

    //h265
#if 0
    const std::string codec("libx265");
    const std::string codec_priv_key = "x265-params";
    const std::string codec_priv_value = "keyint=60:min-keyint=60:scenecut=0";
#else
    //h264 
    const std::string codec("libx264");
    const std::string codec_priv_key = "x264-params";
    const std::string codec_priv_value = "keyint=60:min-keyint=60:scenecut=0:force-cfr=1";
#endif

    self->streamIndex = 0;
    self->pts = 0;
    self->fps = fps;
    self->ptsStep = 12800/fps;

    //
    AVFormatContext *avfc = nullptr;
    avformat_alloc_output_context2(&avfc, 0, 0, fn.c_str());
    HMP_REQUIRE(avfc, "VideoWriter: allocate AVFormatContext failed");
    self->avfc = decltype(self->avfc)(avfc, std::ptr_fun(avformat_free_context));

    AVStream *avs = avformat_new_stream(avfc, NULL);
    self->avs = avs;

    AVCodec *avc = avcodec_find_encoder_by_name(codec.c_str());
    HMP_REQUIRE(avc, "VideoWriter: can't find codec by name {}", codec);

    auto avcc = avcodec_alloc_context3(avc);
    HMP_REQUIRE(avcc, "VideoWriter: allocate avcodec context failed");
    self->avcc = std::shared_ptr<AVCodecContext>(avcc,
         [](AVCodecContext *ptr){ avcodec_free_context(&ptr); });

    av_opt_set(avcc->priv_data, "preset", "fast", 0);
    av_opt_set(avcc->priv_data, "", "fast", 0);
    av_opt_set(avcc->priv_data, codec_priv_key.c_str(), codec_priv_value.c_str(), 0);
    avcc->width = width;
    avcc->height = height;
    avcc->sample_aspect_ratio = av_make_q(1, 1);
    avcc->bit_rate = kbs * 1000;
    avcc->rc_buffer_size = kbs * 2000;
    avcc->rc_max_rate = kbs * 1500;
    avcc->rc_min_rate = kbs * 1000;
    avcc->time_base =  AVRational{1, fps*2};
    avcc->framerate = AVRational{fps, 1};
    avcc->gop_size = 10;
    avcc->max_b_frames = 1;
    assign_pixel_info(*avcc, pix_info);

    //
    avs->time_base = avcc->time_base;

    //
    self->pix_info = pix_info;

    //
    self->avpkt = std::shared_ptr<AVPacket>(av_packet_alloc(), 
        [](AVPacket *pkt) { av_packet_free(&pkt); });
    self->avpkt->stream_index = self->streamIndex;
    self->avpkt->duration = 1;

    HMP_REQUIRE(self->avpkt,
         "FFMPEG: alloc AVPacket or AVFrame failed");

    //
    HMP_REQUIRE(avcodec_open2(avcc, avc, NULL) >= 0, "VideoWriter: open codec failed");
    avcodec_parameters_from_context(avs->codecpar, avcc);

    if(avfc->oformat->flags & AVFMT_GLOBALHEADER){
        avfc->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
    }

    if(!(avfc->oformat->flags & AVFMT_NOFILE)){
        auto rc = avio_open(&avfc->pb, fn.c_str(), AVIO_FLAG_WRITE);
        HMP_REQUIRE(rc >= 0, "VideoWriter: open file({}) to write failed", fn);
    }

    AVDictionary* muxer_opts = NULL;
    HMP_REQUIRE(
        avformat_write_header(avfc, &muxer_opts) >= 0,
        "VideoWriter: write header failed");

}

VideoWriter::~VideoWriter()
{
    //flush
    encodeVideo(self->avfc.get(), self->avcc.get(), NULL, self->avpkt.get());
    av_write_trailer(self->avfc.get());
}


void VideoWriter::write(const std::vector<Frame> &images)
{
    HMP_REQUIRE(images[0].format() == self->pix_info.format(),
         "VideoWriter: invalid pixel format, expect {}, got {}",
         self->pix_info.format(), images[0].format());
    HMP_REQUIRE(images[0].width() == self->avcc->width && 
                images[0].height() == self->avcc->height,
         "VideoWriter: invalid image size");
    HMP_REQUIRE(images[0].plane(0).is_cpu(),
        "VideoWriter: only support CPU images, got {}", 
        stringfy(images[0].plane(0).device()));

    int batch = images.size();
    for(int i = 0; i < batch; ++i){
        //

        auto avframe = std::shared_ptr<AVFrame>(
            to_video_frame(images[i]),
            [](AVFrame *avf){
                    av_frame_free(&avf);
            });

        avframe->pts = self->pts;
        self->pts += self->ptsStep;

        encodeVideo(self->avfc.get(), self->avcc.get(), avframe.get(), self->avpkt.get());
    }
}


}} //namespace hmp::ffmpeg