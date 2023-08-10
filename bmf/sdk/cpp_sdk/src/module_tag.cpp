#include <bmf/sdk/module_tag.h>
#include <exception>
#include <iostream>
#include <map>
#include <type_traits>

namespace bmf_sdk {

ModuleTag operator|(ModuleTag tag1, ModuleTag tag2) {
    if (tag1 == ModuleTag::BMF_TAG_NONE || tag1 == ModuleTag::BMF_TAG_DONE ||
        tag2 == ModuleTag::BMF_TAG_NONE || tag2 == ModuleTag::BMF_TAG_DONE)
        throw std::runtime_error("invalid value");
    return static_cast<ModuleTag>(static_cast<module_tag_type>(tag1) |
                                  static_cast<module_tag_type>(tag2));
}

ModuleTag operator|=(ModuleTag &tag1, ModuleTag tag2) {
    if (tag1 == ModuleTag::BMF_TAG_NONE || tag1 == ModuleTag::BMF_TAG_DONE ||
        tag2 == ModuleTag::BMF_TAG_NONE || tag2 == ModuleTag::BMF_TAG_DONE)
        throw std::runtime_error("invalid value");
    tag1 = static_cast<ModuleTag>(static_cast<module_tag_type>(tag1) |
                                  static_cast<module_tag_type>(tag2));
    return tag1;
}

std::ostream &operator<<(std::ostream &os, const ModuleTag &tag) {
    static std::map<ModuleTag, std::string> m = {
        {ModuleTag::BMF_TAG_DECODER, "BMF_TAG_DECODER"},
        {ModuleTag::BMF_TAG_ENCODER, "BMF_TAG_ENCODER"},
        {ModuleTag::BMF_TAG_FILTER, "BMF_TAG_FILTER"},
        {ModuleTag::BMF_TAG_MUXER, "BMF_TAG_MUXER"},
        {ModuleTag::BMF_TAG_DEMUXER, "BMF_TAG_DEMUXER"},
        {ModuleTag::BMF_TAG_IMAGE_PROCESSOR, "BMF_TAG_IMAGE_PROCESSOR"},
        {ModuleTag::BMF_TAG_AUDIO_PROCESSOR, "BMF_TAG_AUDIO_PROCESSOR"},
        {ModuleTag::BMF_TAG_VIDEO_PROCESSOR, "BMF_TAG_VIDEO_PROCESSOR"},
        {ModuleTag::BMF_TAG_DEVICE_HWACCEL, "BMF_TAG_DEVICE_HWACCEL"},
        {ModuleTag::BMF_TAG_AI, "BMF_TAG_AI"},
        {ModuleTag::BMF_TAG_UTILS, "BMF_TAG_UTILS"},
    };
    std::string str;
    for (const auto &[k, v] : m) {
        if (static_cast<module_tag_type>(tag) &
            static_cast<module_tag_type>(k)) {
            str += (str.size() == 0 ? "" : "|");
            str += v;
        }
    }
    str = (str.size() == 0 ? "BMF_TAG_NONE" : str);
    return os << str;
}

} // namespace bmf_sdk
