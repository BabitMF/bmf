#include "bmf_av_packet.h"
#include <fstream>
#include "bytemeet_video_decoder.h"
using namespace boost::python;

ByteMeetDecode::ByteMeetDecode(int node_id, JsonParam option) : Module(node_id, option) {
    if (option.has_key("video_codec")) {
        option.get_string("video_codec", video_codec_name_);
    }
    if (option.has_key("video_time_base")) {
        option.get_string("video_time_base", video_time_base_);
    }
    codec = avcodec_find_decoder_by_name(video_codec_name_.c_str());

    if (NULL == codec) {
        fprintf(stderr, "Codec not found\n");
    }

    c = avcodec_alloc_context3(codec);
    if (NULL == c) {
        fprintf(stderr, "Could not allocate video codec context\n");
    }
    c->codec_type = AVMEDIA_TYPE_VIDEO;
    AVDictionary *opts = NULL;
    av_dict_set(&opts, "refcounted_frames", "1", 0);
    av_dict_set(&opts, "threads", "auto", 0);
    if (avcodec_open2(c, codec, &opts) < 0) {
        fprintf(stderr, "Could not open codec\n");
    }
}

int ByteMeetDecode::fill_video_packet(AVFrame* frame,Task &task){
    Packet out_pkt;
    out_pkt.set_timestamp(frame->pts);
    if (first_frame_flag_ ){
        av_dict_set(&frame->metadata, "time_base", video_time_base_.c_str(), 0);
        first_frame_flag_ = false;
    }
    VideoFrame video_frame = VideoFrame(frame);

    out_pkt.set_data(video_frame);
    out_pkt.set_data_class_type(VIDEO_FRAME_TYPE);
    out_pkt.set_class_name("libbmf_module_sdk.VideoFrame");
    out_pkt.set_data_type(DATA_TYPE_C);
    task.fill_output_packet(0,out_pkt);
    return 0;
}

int ByteMeetDecode::process(Task &task) {
    int ret = 0, got_frame;
    int index = 0;
    Packet packet;
    while (task.pop_packet_from_input_queue(index, packet)) {
        if (packet.get_timestamp() == BMF_PAUSE){
            task.fill_output_packet(0, packet);
            continue;
        }
        if (packet.get_timestamp() == BMF_EOF) {
            while (1) {
                AVPacket fpkt;
                av_init_packet(&fpkt);
                fpkt.data = NULL;
                fpkt.size = 0;
                AVFrame *frame = av_frame_alloc();
                int decode_len = avcodec_decode_video2(c, frame, &got_frame, &fpkt);
                if (decode_len < 0 || !got_frame) {
                    break;
                }
                fill_video_packet(frame, task);
                av_frame_free(&frame);

            }
            task.fill_output_packet(0, Packet::generate_eof_packet());
            task.set_timestamp(DONE);
            return 0;
        }
        boost::any data = packet.get_data();
        BMFAVPacket bmf_pkt = boost::any_cast<BMFAVPacket>(data);
        AVPacket *av_packet = bmf_pkt.get_av_packet();
        static AVCodecParserContext *avParserContext;
        AVFrame *frame = av_frame_alloc();
        int decode_len = avcodec_decode_video2(c, frame, &got_frame, av_packet);
        if (decode_len < 0)
            fprintf(stderr, "Error while decoding frame %d\n", decode_len);
        else {
            if (got_frame) {
                fill_video_packet(frame, task);
            }
        }
        av_frame_free(&frame);
    }
    return 0;
}

int ByteMeetDecode::close() {
    if (c) {
        avcodec_free_context(&c);
        c = NULL;
    }
    return 0;
}

REGISTER_MODULE_CLASS(ByteMeetDecode)
