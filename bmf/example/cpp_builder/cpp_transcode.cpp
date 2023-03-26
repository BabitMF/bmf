#include "builder.hpp"
#include "bmf_nlohmann/json.hpp"

#include "cpp_test_helper.h"

TEST(cpp_transcode, transcode_simple) {
    std::string output_file = "./simple.mp4";
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
        {"output_path", output_file},
        {"video_params", {
            {"codec", "h264"},
            {"width", 320},
            {"height", 240},
            {"crf", 23},
            {"preset", "veryfast"}
        }},
        {"audio_params", {
            {"codec", "aac"},
            {"bit_rate", 128000},
            {"sample_rate", 44100},
            {"channels", 2}
        }}
    };
    
    graph.Encode(video["video"], video["audio"], bmf_sdk::JsonParam(encode_para));

    graph.Run();

    BMF_CPP_FILE_CHECK(
        output_file, 
        "../transcode/simple.mp4|240|320|7.617000|MOV,MP4,M4A,3GP,3G2,MJ2|369643|351947|h264|{\"fps\": \"30.0662251656\"}"
    );
}

TEST(cpp_transcode, transcode_video) {
    std::string output_file = "./video.mp4";
    BMF_CPP_FILE_REMOVE(output_file);

    bmf_nlohmann::json graph_para = {
        {"dump_graph", 1}
    };

    auto graph = bmf::builder::Graph(bmf::builder::NormalMode, bmf_sdk::JsonParam(graph_para));

    bmf_nlohmann::json decode_tail_para = {
        {"input_path", "../files/header.mp4"}
    };
    auto tail = graph.Decode(bmf_sdk::JsonParam(decode_tail_para));

    bmf_nlohmann::json decode_header_para = {
        {"input_path", "../files/header.mp4"}
    };
    auto header = graph.Decode(bmf_sdk::JsonParam(decode_header_para));

    bmf_nlohmann::json decode_main_para = {
        {"input_path", "../files/img.mp4"}
    };
    auto video = graph.Decode(bmf_sdk::JsonParam(decode_main_para));

    bmf_nlohmann::json decode_logo1_para = {
        {"input_path", "../files/xigua_prefix_logo_x.mov"}
    };
    auto logo_1 = graph.Decode(bmf_sdk::JsonParam(decode_logo1_para))["video"]
        .Scale("320:144");

    bmf_nlohmann::json decode_logo2_para = {
        {"input_path", "../files/xigua_loop_logo2_x.mov"}
    };
    auto logo_2 = graph.Decode(bmf_sdk::JsonParam(decode_logo2_para))["video"]
        .Scale("320:144")
        .Loop("loop=-1:size=991")
        .Setpts("PTS+3.900/TB");

    auto main_video = video["video"].Scale("1280:720")
        .Overlay({logo_1}, "repeatlast=0")
        .Overlay({logo_2}, "shortest=1:x=if(gte(t,3.900),960,NAN):y=0");

    auto concat_video = graph.Concat({
        header["video"].Scale("1280:720"),
        main_video,
        tail["video"].Scale("1280:720")
    }, "n=3");

    auto concat_audio = graph.Concat({
        header["audio"],
        video["audio"],
        tail["audio"]
    }, "a=1:n=3:v=0");

    bmf_nlohmann::json encode_para = {
        {"output_path", output_file},
        {"video_params", {
            {"codec", "h264"},
            {"width", 1280},
            {"height", 720},
            {"crf", 23},
            {"preset", "veryfast"},
            {"x264-params", "ssim=1:psnr=1"},
            {"vsync", "vfr"},
            {"max_fr", 60}
        }},
        {"audio_params", {
            {"codec", "aac"},
            {"bit_rate", 128000},
            {"sample_rate", 44100},
            {"channels", 2}
        }},
        {"mux_params", {
            {"fflags", "+igndts"},
            {"movflags", "+faststart+use_metadata_tags"},
            {"max_interleave_delta", "0"}
        }}
    };

    graph.Encode(concat_video, concat_audio, bmf_sdk::JsonParam(encode_para));

    graph.Run();

    BMF_CPP_FILE_CHECK(
        output_file, 
        "../transcode/video.mp4|720|1280|13.652000|MOV,MP4,M4A,3GP,3G2,MJ2|3902352|6659364|h264|{\"fps\": \"43.29\"}"
    );
}

TEST(cpp_transcode, transcode_image) {
    std::string output_file = "./image.jpg";
    BMF_CPP_FILE_REMOVE(output_file);

    bmf_nlohmann::json graph_para = {
        {"dump_graph", 1}
    };

    auto graph = bmf::builder::Graph(bmf::builder::NormalMode, bmf_sdk::JsonParam(graph_para));

    bmf_nlohmann::json decode_para = {
        {"input_path", "../../example/files/overlay.png"}
    };
    auto video = graph.Decode(bmf_sdk::JsonParam(decode_para));

    bmf_nlohmann::json encode_para = {
        {"output_path", output_file},
        {"format", "mjpeg"},
        {"video_params", {
            {"codec", "jpg"},
            {"width", 320},
            {"height", 240}
        }}
    };
    
    graph.Encode(video["video"].Scale("320:240"), bmf_sdk::JsonParam(encode_para));

    graph.Run();

    BMF_CPP_FILE_CHECK(
        output_file, 
        "image.jpg|240|320|0.040000|IMAGE2|975400|4877|mjpeg|{\"fps\": \"0\"}"
    );
}

TEST(cpp_transcode, transcode_option) {
    std::string output_file = "./option.mp4";
    BMF_CPP_FILE_REMOVE(output_file);

    bmf_nlohmann::json graph_para = {
        {"dump_graph", 1}
    };

    auto graph = bmf::builder::Graph(bmf::builder::NormalMode, bmf_sdk::JsonParam(graph_para));

    bmf_nlohmann::json decode_para = {
        {"input_path", "../../example/files/img.mp4"},
        {"start_time", 2}
    };
    auto video = graph.Decode(bmf_sdk::JsonParam(decode_para));

    bmf_nlohmann::json encode_para = {
        {"output_path", output_file},
        {"video_params", {
            {"codec", "h264"},
            {"width", 1280},
            {"height", 720},
            {"crf", 23},
            {"preset", "fast"},
            {"x264-params", "ssim=1:psnr=1"}
        }},
        {"audio_params", {
            {"codec", "aac"},
            {"bit_rate", 128000},
            {"sample_rate", 44100},
            {"channels", 2}
        }},
        {"mux_params", {
            {"fflags", "+igndts"},
            {"movflags", "+faststart+use_metadata_tags"},
            {"max_interleave_delta", "0"}
        }}
    };
    
    video["video"].EncodeAsVideo(video["audio"], bmf_sdk::JsonParam(encode_para));

    graph.Run();

    BMF_CPP_FILE_CHECK(
        output_file, 
        "../transcode/option.mp4|720|1280|5.643000|MOV,MP4,M4A,3GP,3G2,MJ2|3265125|2303138|h264|{\"fps\": \"30.1796407186\"}"
    );
}
