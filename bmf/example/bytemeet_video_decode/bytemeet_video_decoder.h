#ifndef BMF_BYTEMEET_VIDEO_DECODER_H
#define BMF_BYTEMEET_VIDEO_DECODER_H

#include "common.h"
#include "packet.h"
#include "video_frame.h"
#include "task.h"
#include "module.h"
#include "audio_frame.h"
#include "module_registry.h"
#include <fstream>
#include "bmf_av_packet.h"

USE_BMF_SDK_NS

class ByteMeetDecode : public Module {
public:
    ByteMeetDecode(int node_id, JsonParam option);

    ~ByteMeetDecode() {}

    int fill_video_packet(AVFrame *frame, Task &task);

    virtual int process(Task &task);

    int reset() { return 0; };

    int close();

    bool first_frame_flag_ = true;
    AVCodec *codec = NULL;
    AVCodecContext *c = NULL;
    std::string video_time_base_;
    std::string video_codec_name_;

};

#endif

