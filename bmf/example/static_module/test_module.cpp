#include <stdint.h>
#include <bmf/sdk/module.h>
#include <bmf/sdk/task.h>
#include <bmf/sdk/log.h>
#include <bmf/sdk/video_frame.h>
#include <bmf/sdk/module_registry.h>

using namespace bmf_sdk;

const static char *LOG_TAG = "[VerfastDenoise]";

class VeryfastDenoiseModule : public Module
{
    int cur_width_ = -1;
    int cur_height_ = -1;
    int noise_level_ = 85;

public:
    VeryfastDenoiseModule(int node_id, JsonParam option) 
        : Module(node_id, option)
    {
        noise_level_ = option.has_key("noise_level") ? option.get<float>("noise_level") : 85.f;
    }

    int32_t process(Task &task) override
    {
        Packet packet;
        while (task.pop_packet_from_input_queue(0, packet)) {
            auto ivf = packet.get<VideoFrame>();
            task.fill_output_packet(0, Packet(ivf));
        }

        return 0;
    }
};


REGISTER_MODULE_CLASS(VeryfastDenoiseModule)
