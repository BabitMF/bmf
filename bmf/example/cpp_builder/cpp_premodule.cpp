#include "builder.hpp"
#include "bmf_nlohmann/json.hpp"

#include "cpp_test_helper.h"

TEST(cpp_premodule, premodule) {
    std::string output_file = "./output.mp4";
    BMF_CPP_FILE_REMOVE(output_file);

    bmf_nlohmann::json pre_module_option = {
        {"name", "analysis_SR"},
        {"para", "analysis_SR"}
    };
    auto pre_module = bmf::builder::GetModuleInstance(
        "analysis", 
        pre_module_option.dump(),
        bmf::builder::Python,
        "../../example/pre_module",
        "analysis:analysis"
    );

    for (int i = 0; i < 3; i++) {
        bmf_nlohmann::json graph_para = {
            {"dump_graph", 1}
        };
        bmf_nlohmann::json decode_para = {
            {"input_path", "../../example/files/img.mp4"}
        };
        auto graph = bmf::builder::Graph(bmf::builder::NormalMode, bmf_sdk::JsonParam(graph_para));
        auto video = graph.Decode(bmf_sdk::JsonParam(decode_para));
            
        auto output = video["video"].Scale("320:240");
        auto analyzed = output.PythonModule({}, "analysis", bmf_sdk::JsonParam());
        analyzed.SetPreModule(pre_module);

        bmf_nlohmann::json encode_para = {
            {"output_path", output_file},
            {"video_params", {
                {"width", 300},
                {"height", 200}
            }}
        };
        analyzed.EncodeAsVideo(bmf_sdk::JsonParam(encode_para));
        graph.Run();

        BMF_CPP_FILE_CHECK(output_file, "../pre_module/output.mp4|200|300|7.550000|MOV,MP4,M4A,3GP,3G2,MJ2|208824|197078|h264|{\"fps\": \"30.0662251656\"}");
    }
}
