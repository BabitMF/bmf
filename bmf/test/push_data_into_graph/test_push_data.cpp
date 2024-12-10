/*
#include "../../engine/c_engine/include/common.h"
#include "../../engine/c_engine/include/graph.h"

#include <bmf/sdk/bmf.h>
#include <bmf/sdk/log.h>
#include <fstream>

#include <hmp/imgproc.h>
// #include <iostream>

USE_BMF_ENGINE_NS
USE_BMF_SDK_NS

int main() {
    BMFLOG_SET_LEVEL(BMF_INFO);

    time_t time1 = clock();
    nlohmann::json graph_json;
    std::string config_file = "./original_graph.json";
    std::ifstream gs(config_file);
    gs >> graph_json;
    GraphConfig graph_config(graph_json);
    std::map<int, std::shared_ptr<Module>> pre_modules;
    std::map<int, std::shared_ptr<ModuleCallbackLayer>> callback_bindings;
    std::shared_ptr<Graph> graph =
        std::make_shared<Graph>(graph_config, pre_modules, callback_bindings);
    std::cout << "init graph success" << std::endl;

    graph->start();

    int count = 50;
    int pts_ = 0;
    while(count > 0) {
        auto frame = VideoFrame(640, 480, PixelInfo(PixelFormat::PF_YUV420P));
        frame.set_pts(pts_);
        pts_++;
        auto packet = Packet(frame);
        count--;
        graph->add_input_stream_packet("outside_raw_video", packet);
        std::cout << "push data into inputstream" << std::endl;
    }
    auto packet = Packet::generate_eof_packet();
    graph->add_input_stream_packet("outside_raw_video", packet);

    graph->close();
    time_t time2 = clock();
    std::cout << "time:" << time2 - time1 << std::endl;

    return 0;
}
*/

#include <builder.hpp>
#include "nlohmann/json.hpp"
#include <bmf/sdk/bmf.h>

int main() {
    std::string output_path = "./push_data_output.mp4";

    auto graph = bmf::builder::Graph(bmf::builder::PushDataMode);
    auto video_stream = graph.InputStream("outside_raw_video", "", "");
    nlohmann::json encode_para = {
        {"output_path", output_path},
        {"video_params", {
            {"codec", "h264"},
            {"width", 640},
            {"height", 480},
            {"crf", 23},
            {"preset", "veryfast"},
            {"vsync", "vfr"}
            }
        }
    };
    auto video = graph.Encode(video_stream, bmf_sdk::JsonParam(encode_para));

    graph.Start();

    int count = 50;
    int pts_ = 0;
    while (count > 0) {
        auto frame = VideoFrame(640, 480, PixelInfo(PixelFormat::PF_YUV420P));
        frame.set_pts(pts_);
        pts_++;
        auto packet = Packet(frame);
        count--;
        graph.FillPacket(video_stream.GetName(), packet);
        std::cout << "push data into inputstream" << std::endl;
    }
    auto packet = Packet::generate_eof_packet();
    graph.FillPacket(video_stream.GetName(), packet);

    graph.Close();

    return 0;
}