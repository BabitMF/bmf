#ifndef BMF_OPUS_DECODER_MODULE_H
#define BMF_OPUS_DECODER_MODULE_H

#include <bmf/sdk/common.h>
#include <bmf/sdk/packet.h>
#include <bmf/sdk/task.h>
#include <bmf/sdk/module.h>
#include <bmf/sdk/module_registry.h>

#include "fraction.hpp"

#include <chrono>

USE_BMF_SDK_NS

class ClockModule : public Module {
public:
    ClockModule(int node_id, JsonParam option);

    ~ClockModule() {}

    int process(Task &task);

    int reset() { return 0; };

    bool is_hungry(int input_stream_id);

private:
    Fraction::Fraction fps_tick_, time_base_;
    uint64_t frm_cnt_;

    std::chrono::high_resolution_clock::duration tick_;
    std::chrono::time_point<std::chrono::high_resolution_clock> lst_ts_;
};

#endif

