#ifndef BMF_PULL_STREAM_MODULE_H
#define BMF_PULL_STREAM_MODULE_H

#include "common.h"

#include "packet.h"
#include "video_frame.h"
#include "task.h"
#include "module.h"
#include "audio_frame.h"
#include "module_registry.h"
#include <fstream>

USE_BMF_SDK_NS


struct DataSource{
    std::string data_path;
    std::string size_path;
    std::ifstream f_data;
    std::ifstream f_size;
};

class PullAudioStreamModule : public Module
{
public:
    PullAudioStreamModule(int node_id,JsonParam option);

    ~PullAudioStreamModule() { }

    virtual int process(Task &task);
    std::string input_path_;
    std::vector<DataSource*> source_list;
    AVFormatContext *input_fmt_ctx_;
    std::string pts_file_path_;
    int stream_index_;
    int64_t pts_=28774-66;
    int timestamp_ = 0;
};

#endif

