#include "builder.hpp"
#include "bmf_nlohmann/json.hpp"

void rgb2video() {
    bmf_nlohmann::json graph_para = {
        {"dump_graph", 1}
    };
    auto graph = bmf::builder::Graph(bmf::builder::NormalMode, bmf_sdk::JsonParam(graph_para));

    bmf_nlohmann::json decode_para = {
        {"input_path", "/opt/tiger/bmf/example/files/test_rgba_806x654.rgb"},
        {"s", "806:654"},
        {"pix_fmt", "rgba"}
    };
    auto stream = graph.Decode(bmf_sdk::JsonParam(decode_para));

    auto video_stream = stream["video"].FFMpegFilter({}, "loop", "loop=50:size=1");

    bmf_nlohmann::json encode_para = {
        {"output_path", "./rgb2video.mp4"},
    };

    // graph.Encode(video_stream, bmf_sdk::JsonParam(encode_para));

    graph.Encode(video_stream, stream["audio"], bmf_sdk::JsonParam(encode_para));

    graph.Run();
}
