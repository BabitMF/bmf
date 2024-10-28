#include "builder.hpp"
#include "nlohmann/json.hpp"
#include <cstdio>
#include <iostream>
#include <string.h>
#include <unistd.h>

int main() {
  bmf::builder::Graph graph = bmf::builder::Graph(bmf::builder::NormalMode);

  // create decoder
  nlohmann::json decoder_option = {{"input_path", "big_bunny_10s_30fps.mp4"}};
  auto decoder = graph.Sync(std::vector<int>{}, std::vector<int>{0},
                            decoder_option, "c_ffmpeg_decoder");

  auto copy = graph.Sync(std::vector<int>{0}, std::vector<int>{0},
                         bmf_sdk::JsonParam(decoder_option), "CopyModule",
                         bmf::builder::CPP, "lib/libcopy_module.so",
                         "copy_module:CopyModule");
  nlohmann::json encoder_option = {
      {"output_path", "video_copied.mp4"}};
  auto encoder = graph.Sync(std::vector<int>{0}, std::vector<int>{},
                            encoder_option, "c_ffmpeg_encoder");
  graph.Init(decoder);
  graph.Init(copy);
  graph.Init(encoder);

  // decode
  auto decoded_frames = decoder.ProcessPkts();

  // copy
  bmf::builder::SyncPackets input_copy;
  input_copy.Insert(0, decoded_frames[0]);
  auto copy_frames = copy.ProcessPkts(input_copy);

  // encode
  bmf::builder::SyncPackets input_encode;
  input_encode.Insert(0, copy_frames[0]);
  encoder.ProcessPkts(input_encode);

  encoder.SendEOF();
  return 0;
}