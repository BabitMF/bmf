#ifndef _TOKENIZER_H
#define _TOKENIZER_H
#include <unordered_map>
#include <string>
#include <cstdint>
#include <vector>
#include <re2/re2.h>
#include <regex>
#include <optional>

class tokenizer
{
public:
    tokenizer();
    tokenizer(std::string &path);
    std::vector<int32_t> encode(const std::string &prompts, int32_t max_size);
    std::string decode(std::vector<int32_t> &token_ids);
    bool is_special_id(int32_t id);
    std::pair<std::optional<std::string>, re2::StringPiece> split_special_tokens(re2::StringPiece &input);

private:
    std::unordered_map<std::string, int32_t> encoder_map;
    std::unordered_map<int32_t, std::string> decoder_map;
    std::unordered_map<std::string, int> special_encoder_map;
    std::unique_ptr<re2::RE2> normal_regex;
    std::unique_ptr<re2::RE2> special_regex;
    std::vector<int32_t> special_ids;
    std::string normal_pattern = R"((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?:$|[^\S])|\s+)";
};

#endif