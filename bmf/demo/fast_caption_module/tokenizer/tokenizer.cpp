/*
    MIT License

    Copyright (c) 2023 Jiahao Li

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
*/
#include "tokenizer.h"
#include <fstream>
#include <optional>
#include <algorithm>
#include <bmf/sdk/log.h>
#include "nlohmann/json.hpp"

using json = nlohmann::json;
tokenizer::tokenizer() = default;
tokenizer::tokenizer(std::string &path)
{
    std::ifstream file(path);
    if (!file)
    {
        BMFLOG(BMF_ERROR) << "no file exist in " << path;
    }
    json tokenizer_json;
    file >> tokenizer_json;
    if (tokenizer_json.find("added_tokens") != tokenizer_json.end())
    {
        auto added_tokens = tokenizer_json["added_tokens"];
        for (auto token : added_tokens)
        {
            int id = token["id"];
            std::string content = token["content"];
            special_encoder_map[content] = id;
        }
    }
    std::string tmp{(char)255};
    if (tokenizer_json.find("model") != tokenizer_json.end())
    {
        auto model = tokenizer_json["model"];
        auto vocab = model["vocab"];
        for (auto v : vocab.items())
        {
            std::string key = v.key();
            int val = v.value();
            if (val > 2 && val < 36)
            {
                key = (char)(val - 3);
            }
            auto vocab_iter = key.find("\xe2\x96\x81");
            while (vocab_iter != std::string::npos)
            {
                key.replace(vocab_iter, 3, tmp);
                vocab_iter = key.find("\xe2\x96\x81");
            }
            encoder_map[key] = val;
            decoder_map[val] = key;
        }
    }
    // normal_regex
    normal_regex = std::make_unique<re2::RE2>("(" + normal_pattern + ")");
    // special_regex
    std::string special_pattern;
    for (const auto &item : special_encoder_map)
    {
        special_pattern += re2::RE2::QuoteMeta(item.first) + "|";
        special_ids.emplace_back(item.second);
    }
    if (special_pattern.empty())
    {
        special_regex = nullptr;
    }
    else
    {
        special_pattern.pop_back();
        special_regex = std::make_unique<re2::RE2>("(" + special_pattern + ")");
    }
}

std::pair<std::optional<std::string>, re2::StringPiece> tokenizer::split_special_tokens(re2::StringPiece &input)
{
    if (special_regex == nullptr)
        return {std::nullopt, input};

    auto start = input.begin();
    std::string special;

    while (true)
    {
        if (!re2::RE2::FindAndConsume(&input, *special_regex, &special))
        {
            break;
        }

        if (special_encoder_map.count(special) == 1)
        {
            return {std::move(special), re2::StringPiece(&*start, input.begin() - start - special.size())};
        }
    }
    return {std::nullopt, input};
}

std::vector<std::pair<size_t, int>> _byte_pair_merge(const std::unordered_map<std::string, int> &ranks,
                                                     const std::string &piece)
{
    using rank_t = int;

    std::vector<std::pair<size_t, rank_t>> parts; // (start, rank)
    parts.reserve(piece.length() + 1);

    auto min_rank = std::make_pair<rank_t, size_t>(std::numeric_limits<rank_t>::max(),
                                                   std::numeric_limits<size_t>::max()); // (rank, start)

    for (size_t i = 0; i < piece.length() - 1; i++)
    {
        rank_t rank = std::numeric_limits<rank_t>::max();
        if (auto it = ranks.find(piece.substr(i, 2)); it != ranks.end())
        {
            rank = it->second;
        }
        if (rank < min_rank.first)
        {
            min_rank = std::make_pair(rank, i);
        }
        parts.emplace_back(std::make_pair(i, rank));
    }
    parts.emplace_back(std::make_pair(piece.length() - 1, std::numeric_limits<rank_t>::max()));
    parts.emplace_back(std::make_pair(piece.length(), std::numeric_limits<rank_t>::max()));

    auto get_rank = [&piece, &ranks](const std::vector<std::pair<size_t, rank_t>> &parts, size_t i)
    {
        if (i + 3 < parts.size())
        {
            size_t start = parts[i].first;
            size_t end = parts[i + 3].first;
            if (auto it = ranks.find(piece.substr(start, end - start)); it != ranks.end())
            {
                return it->second;
            }
        }
        return std::numeric_limits<rank_t>::max();
    };

    while (min_rank.first != std::numeric_limits<rank_t>::max())
    {
        size_t i = min_rank.second;
        if (i > 0)
        {
            parts[i - 1].second = get_rank(parts, i - 1);
        }
        parts[i].second = get_rank(parts, i);
        parts.erase(parts.begin() + i + 1);

        min_rank = std::make_pair(std::numeric_limits<rank_t>::max(), std::numeric_limits<size_t>::max());
        for (size_t i = 0; i < parts.size() - 1; i++)
        {
            rank_t rank = parts[i].second;
            if (rank < min_rank.first)
            {
                min_rank = std::make_pair(rank, i);
            }
        }
    }

    return parts;
}

std::vector<int> byte_pair_encode(const std::string &piece,
                                  const std::unordered_map<std::string, int> &ranks)
{
    auto parts = _byte_pair_merge(ranks, piece);

    std::vector<int> tokens;
    tokens.reserve(parts.size() - 1);

    for (size_t i = 1; i < parts.size(); i++)
    {
        size_t start = parts[i - 1].first;
        size_t end = parts[i].first;
        int rank = ranks.at(piece.substr(start, end - start));
        tokens.emplace_back(rank);
    }

    return tokens;
}

std::vector<int32_t> tokenizer::encode(const std::string &prompt, int32_t max_size)
{
    re2::StringPiece input(prompt);
    std::vector<int32_t> output;
    std::string tmp{(char)255};
    while (true)
    {
        auto [special, sub_input] = split_special_tokens(input);
        std::string piece;
        while (re2::RE2::FindAndConsume(&sub_input, *normal_regex, &piece))
        {
            auto space_iter = piece.find(" ");
            while (space_iter != std::string::npos)
            {
                piece.replace(space_iter, 1, tmp);
                space_iter = piece.find(" ");
            }
            auto iter = encoder_map.find(piece);
            if (iter != encoder_map.end())
            {
                // last_piece_token_len = 1;
                output.push_back(iter->second);
                continue;
            }
            auto size = piece.size();
            auto tokens = byte_pair_encode(piece, encoder_map);
            // last_piece_token_len = tokens.size();
            output.insert(output.end(), tokens.begin(), tokens.end());
        }
        if (special)
        {
            int token = special_encoder_map.at(*special);
            output.push_back(token);
        }
        else
        {
            break;
        }
    }
    if (output.size() > max_size)
    {
        output.erase(output.begin() + max_size, output.end());
    }
    return output;
}

bool tokenizer::is_special_id(int32_t id)
{
    return std::find(special_ids.begin(), special_ids.end(), id) != special_ids.end();
}

std::string tokenizer::decode(std::vector<int32_t> &token_ids)
{
    token_ids.erase(std::remove_if(token_ids.begin(), token_ids.end(),
                                   [this](int32_t id)
                                   { return is_special_id(id); }),
                    token_ids.end());
    std::string output;
    output.reserve(token_ids.size() * 2);
    for (auto token : token_ids)
    {
        std::string token_bytes;
        auto iter = decoder_map.find(token);
        std::string bytes = iter->second;
        auto size = bytes.size();
        int32_t i = 0;
        for (; i < size; i++)
        {
            if (bytes[i] == '\377')
            {
                token_bytes += " ";
            }
            else
            {
                token_bytes += bytes[i];
            }
        }
        output += token_bytes;
    }
    return output;
}
