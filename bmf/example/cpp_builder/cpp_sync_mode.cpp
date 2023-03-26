#include "builder.hpp"
#include "bmf_nlohmann/json.hpp"

#include "cpp_test_helper.h"

TEST(cpp_sync_mode, sync_videoframe) {
    std::string output_file = "./videoframe.jpg";
    BMF_CPP_FILE_REMOVE(output_file);

    bmf::builder::Graph graph = bmf::builder::Graph(bmf::builder::NormalMode);

    // create decoder
    bmf_nlohmann::json decoder_option = {
        {"input_path", "../../example/files/overlay.png"}
    };
    auto decoder = graph.Sync(std::vector<int> {}, std::vector<int> {0}, 
        bmf_sdk::JsonParam(decoder_option), "c_ffmpeg_decoder");

    // create scale
    bmf_nlohmann::json scale_option = {
        {"name", "scale"},
        {"para", "320:240"}
    };
    auto scale = graph.Sync(std::vector<int> {0}, std::vector<int> {0}, 
        bmf_sdk::JsonParam(scale_option), "c_ffmpeg_filter");

    // create encoder
    bmf_nlohmann::json encoder_option = {
        {"output_path", output_file},
        {"format", "mjpeg"},
        {"video_params", {
            {"codec", "jpg"}
        }}
    };
    auto encoder = graph.Sync(std::vector<int> {0}, std::vector<int> {}, 
        bmf_sdk::JsonParam(encoder_option), "c_ffmpeg_encoder");

    // call init if necessary, otherwise we skip this step
    graph.Init(decoder);
    graph.Init(scale);
    graph.Init(encoder);

    // decode
    auto decoded_frames = decoder.ProcessPkts();

    // scale
    bmf::builder::SyncPackets input_scale;
    input_scale.Insert(0, decoded_frames[0]);
    auto scaled_frames = scale.ProcessPkts(input_scale);

    // encode
    bmf::builder::SyncPackets input_encode;
    input_encode.Insert(0, scaled_frames[0]);
    encoder.ProcessPkts(input_encode);

    encoder.SendEOF();

    // call close if necessary, otherwise we skip this step
    graph.Close(decoder);
    graph.Close(scale);
    graph.Close(encoder);

    BMF_CPP_FILE_CHECK(output_file, "./videoframe.jpg|240|320|0.04|IMAGE2|950000|4750|mjpeg|{\"fps\": \"0.0\"}");
}

TEST(cpp_sync_mode, sync_audioframe) {
    std::string output_file = "./audioframe.mp4";
    BMF_CPP_FILE_REMOVE(output_file);

    bmf::builder::Graph graph = bmf::builder::Graph(bmf::builder::NormalMode);

    // create decoder
    bmf_nlohmann::json decoder_option = {
        {"input_path", "../../example/files/img.mp4"}
    };
    auto decoder = graph.Sync(std::vector<int> {}, std::vector<int> {1}, decoder_option, "c_ffmpeg_decoder");

    // create volume
    bmf_nlohmann::json volume_option = {
        {"name", "volume"},
        {"para", "volume=3"}
    };
    auto volume = graph.Sync(std::vector<int> {0}, std::vector<int> {0}, volume_option, "c_ffmpeg_filter");

    // create encoder
    bmf_nlohmann::json encoder_option = {
        {"output_path", output_file}
    };
    auto encoder = graph.Sync(std::vector<int> {0,1}, std::vector<int> {}, encoder_option, "c_ffmpeg_encoder");

    // decode and get audio frame
    auto decoded_frames = decoder.ProcessPkts(std::map<int, std::vector<Packet> > {});

    // volume
    std::map<int, std::vector<Packet> > input_volume;
    input_volume.insert(std::make_pair(0, decoded_frames[1]));
    auto volume_frames = volume.ProcessPkts(input_volume);

    // encode
    std::map<int, std::vector<Packet> > input_encode;
    input_encode.insert(std::make_pair(1, input_volume[0]));
    encoder.ProcessPkts(input_encode);

    // send eof to encoder
    encoder.SendEOF();

    BMF_CPP_FILE_CHECK(output_file, "./audioframe.mp4|0|0|0.047|MOV,MP4,M4A,3GP,3G2,MJ2|135319|795||{}");
}

TEST(cpp_sync_mode, sync_video) {
    std::string output_file = "./video.mp4";
    BMF_CPP_FILE_REMOVE(output_file);

    bmf::builder::Graph graph = bmf::builder::Graph(bmf::builder::NormalMode);

    // create sync modules
    bmf_nlohmann::json decoder_option = {
        {"input_path", "../../example/files/img.mp4"}
    };
    auto decoder = graph.Sync(std::vector<int> {}, std::vector<int> {0,1}, decoder_option, "c_ffmpeg_decoder");

    bmf_nlohmann::json scale_option = {
        {"name", "scale"},
        {"para", "320:250"}
    };
    auto scale = graph.Sync(std::vector<int> {0}, std::vector<int> {0}, 
        bmf_sdk::JsonParam(scale_option), "c_ffmpeg_filter");

    bmf_nlohmann::json volume_option = {
        {"name", "volume"},
        {"para", "volume=3"}
    };
    auto volume = graph.Sync(std::vector<int> {0}, std::vector<int> {0}, volume_option, "c_ffmpeg_filter");

    bmf_nlohmann::json encoder_option = {
        {"output_path", output_file}
    };
    auto encoder = graph.Sync(std::vector<int> {0,1}, std::vector<int> {}, encoder_option, "c_ffmpeg_encoder");

    // call init if necessary, otherwise we skip this step
    graph.Init(decoder);
    graph.Init(scale);
    graph.Init(volume);
    graph.Init(encoder);

    // process video/audio by sync mode
    while (1) {
        auto decoded_frames = decoder.ProcessPkts();
        bool has_next = false;
        for (const auto &stream : decoded_frames.packets) {
            if (!stream.second.empty()) {
                has_next = true;
                if (stream.first == 0) {
                    bmf::builder::SyncPackets input_scale;
                    input_scale.Insert(0, decoded_frames[0]);
                    auto scaled_frames = scale.ProcessPkts(input_scale);

                    bmf::builder::SyncPackets input_encoder;
                    input_encoder.Insert(0, scaled_frames[0]);
                    encoder.ProcessPkts(input_encoder);
                } else if (stream.first == 1) {
                    bmf::builder::SyncPackets input_volume;
                    input_volume.Insert(0, decoded_frames[1]);
                    auto volume_frames = volume.ProcessPkts(input_volume);

                    bmf::builder::SyncPackets input_encoder;
                    input_encoder.Insert(1, volume_frames[0]);
                    encoder.ProcessPkts(input_encoder);
                }
            }
        }
        if (!has_next) {
            break;
        }
    }

    // call close if necessary, otherwise we skip this step
    graph.Close(decoder);
    graph.Close(scale);
    graph.Close(volume);
    graph.Close(encoder);

    BMF_CPP_FILE_CHECK(output_file, "./video.mp4|250|320|7.617|MOV,MP4,M4A,3GP,3G2,MJ2|418486|398451|h264|{\"fps\": \"30\"}");
}

TEST(cpp_sync_mode, sync_audio) {
    std::string output_file = "./audio.mp4";
    BMF_CPP_FILE_REMOVE(output_file);

    bmf::builder::Graph graph = bmf::builder::Graph(bmf::builder::NormalMode);

    // create sync modules
    bmf_nlohmann::json decoder_option = {
        {"input_path", "../../example/files/img.mp4"}
    };
    auto decoder = graph.Sync(std::vector<int> {}, std::vector<int> {1}, decoder_option, "c_ffmpeg_decoder");

    bmf_nlohmann::json volume_option = {
        {"name", "volume"},
        {"para", "volume=3"}
    };
    auto volume = graph.Sync(std::vector<int> {0}, std::vector<int> {0}, volume_option, "c_ffmpeg_filter");

    bmf_nlohmann::json encoder_option = {
        {"output_path", output_file}
    };
    auto encoder = graph.Sync(std::vector<int> {0,1}, std::vector<int> {}, encoder_option, "c_ffmpeg_encoder");
    
    // process video/audio by sync mode
    while (1) {
        auto decoded_frames = decoder.ProcessPkts();
        if (decoded_frames[1].empty()) {
            encoder.SendEOF();
            break;
        }

        bmf::builder::SyncPackets input_volume;
        input_volume.Insert(0, decoded_frames[1]);
        auto volume_frames = volume.ProcessPkts(input_volume);
        
        bmf::builder::SyncPackets input_encoder;
        input_encoder.Insert(1, volume_frames[0]);
        encoder.ProcessPkts(input_encoder);
    }

    BMF_CPP_FILE_CHECK(output_file, "./audio.mp4|0|0|7.617|MOV,MP4,M4A,3GP,3G2,MJ2|131882|125569||{}");
}
