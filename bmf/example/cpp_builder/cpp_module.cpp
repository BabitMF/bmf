#include "builder.hpp"
#include "bmf_nlohmann/json.hpp"

#include "cpp_test_helper.h"

TEST(cpp_modules, module_python) {
    std::string output_file = "./output.mp4";
    BMF_CPP_FILE_REMOVE(output_file);

    bmf_nlohmann::json graph_para = {
        {"dump_graph", 1}
    };
    auto graph = bmf::builder::Graph(bmf::builder::NormalMode, bmf_sdk::JsonParam(graph_para));

    bmf_nlohmann::json decode_para = {
        {"input_path", "../../example/files/img.mp4"}
    };
    auto video = graph.Decode(bmf_sdk::JsonParam(decode_para));

    bmf_nlohmann::json encode_para = {
        {"output_path", output_file}
    };

    graph.Module({video["video"]}, "my_module", bmf::builder::Python, bmf_sdk::JsonParam(),
            "MyModule", "../../example/customize_module", "my_module:my_module"
        ).EncodeAsVideo(video["audio"], bmf_sdk::JsonParam(encode_para));

    graph.Run();

    BMF_CPP_FILE_CHECK(
        output_file, 
        "|1080|1920|7.615000|MOV,MP4,M4A,3GP,3G2,MJ2|4483427|4267663|h264|{\"fps\": \"30.0662251656\"}"
    );
}

TEST(cpp_modules, module_cpp) {
    std::string output_file = "./output.mp4";
    BMF_CPP_FILE_REMOVE(output_file);

    bmf_nlohmann::json graph_para = {
        {"dump_graph", 1}
    };
    auto graph = bmf::builder::Graph(bmf::builder::NormalMode, bmf_sdk::JsonParam(graph_para));

    bmf_nlohmann::json decode_para = {
        {"input_path", "../../example/files/img.mp4"}
    };
    auto video = graph.Decode(bmf_sdk::JsonParam(decode_para));

    auto video_2 = graph.Module(
        {video["video"]}, 
        "copy_module", 
        bmf::builder::CPP, 
        bmf_sdk::JsonParam(), 
        "CopyModule", 
        "../lib/libcopy_module.so", 
        "copy_module:CopyModule"
    );

    bmf_nlohmann::json encode_para = {
        {"output_path", output_file},
        {"video_params", {
            {"vsync", "vfr"},
            {"max_fr", 60}
        }},
        {"audio_params", {
            {"codec", "aac"}
        }}
    };
    graph.Encode(video_2, video["audio"], bmf_sdk::JsonParam(encode_para));

    graph.Run();

    BMF_CPP_FILE_CHECK(
        output_file, 
        "../c_module/output.mp4|1080|1920|7.615000|MOV,MP4,M4A,3GP,3G2,MJ2|4483410|4267646|h264|{\"fps\": \"30.0662251656\"}"
    );
}

TEST(cpp_modules, audio_python_module) {
    std::string output_file = "./audio_python_module";
    BMF_CPP_FILE_REMOVE(output_file);

    bmf_nlohmann::json graph_para = {
        {"dump_graph", 1}
    };
    auto graph = bmf::builder::Graph(bmf::builder::NormalMode, bmf_sdk::JsonParam(graph_para));

    bmf_nlohmann::json decode_para = {
        {"input_path", "../../example/files/img.mp4"}
    };
    auto audio = graph.Decode(bmf_sdk::JsonParam(decode_para))["audio"];

    bmf_nlohmann::json encode_para = {
        {"output_path", output_file}
    };

    auto audio_output = graph.Module({audio}, "my_module", bmf::builder::Python, bmf_sdk::JsonParam(),
            "MyModule", "../../example/customize_module", "my_module:my_module"
        );
    graph.Encode(graph.NewPlaceholderStream(), audio_output, bmf_sdk::JsonParam(encode_para));
    graph.Run();

    BMF_CPP_FILE_CHECK(
        output_file, 
        "../audio_copy/audio_c_module.mp4|0|0|7.617000|MOV,MP4,M4A,3GP,3G2,MJ2|136031|129519||{}"
    );
}

TEST(cpp_modules, test_exception_in_python_module) {
    std::string output_file = "./test_exception_in_python_module.mp4";

    bmf_nlohmann::json graph_para = {
        {"dump_graph", 1}
    };
    auto graph = bmf::builder::Graph(bmf::builder::NormalMode, bmf_sdk::JsonParam(graph_para));

    bmf_nlohmann::json decode_para = {
        {"input_path", "../../example/files/img.mp4"}
    };
    auto audio = graph.Decode(bmf_sdk::JsonParam(decode_para))["audio"];

    bmf_nlohmann::json encode_para = {
        {"output_path", output_file}
    };
    bmf_nlohmann::json module_para = {
        {"exception", 1}
    };

    auto audio_output = graph.Module({audio}, "my_module", bmf::builder::Python, bmf_sdk::JsonParam(module_para),
            "MyModule", "../../example/customize_module", "my_module:my_module"
        );
    try
    {
        graph.Encode(graph.NewPlaceholderStream(), audio_output, bmf_sdk::JsonParam(encode_para));
        graph.Run();
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << '\n';
    }
}
