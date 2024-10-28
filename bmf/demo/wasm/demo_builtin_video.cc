#include "builder.hpp"
#include "nlohmann/json.hpp"
#include <cstdio>
#include <iostream>
#include <string.h>
#include <unistd.h>

int main() {
  bmf::builder::Graph graph = bmf::builder::Graph(bmf::builder::NormalMode);

  nlohmann::json decoder_option = {{"input_path", "big_bunny_10s_30fps.mp4"}};
  auto decoder = graph.Sync(std::vector<int>{}, std::vector<int>{0, 1},
                            decoder_option, "c_ffmpeg_decoder");

  nlohmann::json scale_option = {{"name", "scale"}, {"para", "320:250"}};
  auto scale = graph.Sync(std::vector<int>{0}, std::vector<int>{0},
                          bmf_sdk::JsonParam(scale_option), "c_ffmpeg_filter");

  nlohmann::json volume_option = {{"name", "volume"}, {"para", "volume=3"}};
  auto volume = graph.Sync(std::vector<int>{0}, std::vector<int>{0},
                           volume_option, "c_ffmpeg_filter");

  nlohmann::json encoder_option = {{"output_path", "video_simple_interface.mp4"}};
  auto encoder = graph.Sync(std::vector<int>{0, 1}, std::vector<int>{},
                            encoder_option, "c_ffmpeg_encoder");
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
  BMFLOG(BMF_INFO) << "done" << std::endl;
  return 0;
}