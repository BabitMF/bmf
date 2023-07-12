#pragma once

#include <bmf/sdk/common.h>
#include <ostream>

namespace bmf_sdk {

using module_tag_type = int64_t;

enum class BMF_API ModuleTag : module_tag_type {
  BMF_TAG_NONE = 0x0,

  BMF_TAG_DECODER = 0x01 << 0,
  BMF_TAG_ENCODER = 0x01 << 1,
  BMF_TAG_FILTER = 0x01 << 2,
  BMF_TAG_MUXER = 0x01 << 3,
  BMF_TAG_DEMUXER = 0x01 << 4,
  BMF_TAG_IMAGE_PROCESSOR = 0x01 << 5,
  BMF_TAG_AUDIO_PROCESSOR = 0x01 << 6,
  BMF_TAG_VIDEO_PROCESSOR = 0x01 << 7,
  BMF_TAG_DEVICE_HWACCEL = 0x01 << 8,
  BMF_TAG_AI = 0x01 << 9,
  BMF_TAG_UTILS = 0x01 << 10,

  BMF_TAG_DONE = 0x01LL << (sizeof(module_tag_type) * 8 - 1),
};

ModuleTag BMF_API operator|(ModuleTag tag1, ModuleTag tag2);
ModuleTag BMF_API operator|=(ModuleTag &tag1, ModuleTag tag2);
std::ostream BMF_API &operator<<(std::ostream &os, const ModuleTag &tag);

} // namespace bmf_sdk
