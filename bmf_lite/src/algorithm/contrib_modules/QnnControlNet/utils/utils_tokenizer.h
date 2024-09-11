#include <regex>
using namespace std;
#include <sstream>
#include <fstream>
#include <sys/stat.h>
#include <set>

#include "utils/log.h"
/*================================================== CLIPTokenizer
 * ===================================================*/

// Ref: https://github.com/openai/CLIP/blob/main/clip/simple_tokenizer.py
// TODO: implement bpe
class CLIPTokenizer {
  private:
    int is_ch_ = 0;

    int UNK_TOKEN_ID = 49407;
    int BOS_TOKEN_ID = 49406;
    int EOS_TOKEN_ID = 49407;
    int PAD_TOKEN_ID = 49407;

    map<string, int> tokenizer_token2idx;
    map<int, string> tokenizer_idx2token;
    std::map<std::pair<std::string, std::string>, int> bpeRanks;

    static std::string strip(const std::string &str) {
        std::string::size_type start = str.find_first_not_of(" \t\n\r\v\f");
        std::string::size_type end = str.find_last_not_of(" \t\n\r\v\f");

        if (start == std::string::npos) {
            // String contains only whitespace characters
            return "";
        }

        return str.substr(start, end - start + 1);
    }

    static std::string whitespace_clean_and_low_case(std::string text) {
        text = std::regex_replace(text, std::regex(R"(\s+)"), " ");
        text = strip(text);
        std::transform(text.begin(), text.end(), text.begin(),
                       [](unsigned char c) { return std::tolower(c); });
        return text;
    }

    std::set<std::pair<std::string, std::string>>
    getPairs(const std::vector<std::string> &word) {
        std::set<std::pair<std::string, std::string>> pairs;
        for (size_t i = 0; i < word.size() - 1; ++i) {
            pairs.insert({word[i], word[i + 1]});
        }
        return pairs;
    };

    std::map<std::pair<std::string, std::string>, int>
    loadBpeRanks(const std::string &merge_path) {
        std::map<std::pair<std::string, std::string>, int> result;
        std::ifstream reader(merge_path);
        std::string line;

        int startLine = 1;
        int count = 0;
        while (std::getline(reader, line)) {
            // remove first line
            if (startLine != 0 && startLine-- > 0)
                continue;
            std::string first, second;
            std::istringstream iss(line);
            if (iss >> first >> second) {
                result[std::make_pair(first, second)] = count++;
            }
        }
        reader.close();

        return result;
    }

    std::vector<std::string> str_bpe_ch(const std::string &token) {
        if (token.empty())
            return {token};

        std::vector<std::string> word;
        for (size_t i = 0; i < token.length();) {
            int cplen = 1;
            if ((token[i] & 0xf8) == 0xf0)
                cplen = 4; // 占用4个字节，前5位为11110
            else if ((token[i] & 0xf0) == 0xe0)
                cplen = 3; // 占用3个字节，前4位为1110
            else if ((token[i] & 0xe0) == 0xc0)
                cplen = 2; // 占用2个字节，前3位为110
            // 个人感觉这行代码好像没什么用，如果三种情况都不符合，那么cplen就为初始化的0，是符合utf-8编码定义的
            if ((i + cplen) > token.length())
                cplen = 1;
            auto sub_str = token.substr(i, cplen);
            word.push_back(sub_str);
            i += cplen;
        }

        return word;
    }

    std::vector<std::string> str_bpe_en(const std::string &token) {
        if (token.empty())
            return {token};

        std::vector<std::string> word;

        auto token_word = token + "</w>";

        if (tokenizer_token2idx.find(token_word) != tokenizer_token2idx.end()) {
            word.push_back(token_word);
            return word;
        }

        for (char c : token) {
            word.emplace_back(1, c);
        }
        word.back() += "</w>";

        while (true) {
            std::pair<std::string, std::string> min;
            int minValue = 0;
            bool minFound = false;
            for (const auto &pair : getPairs(word)) {
                auto it = bpeRanks.find(pair);
                if (it == bpeRanks.end())
                    continue;
                // todo check bpeRanks count compare
                if (!minFound || it->second < minValue) {
                    min = pair;
                    minValue = it->second;
                    minFound = true;
                }
            }

            if (!minFound)
                break;

            std::vector<std::string> newWord;
            for (size_t i = 0; i < word.size();) {
                if (i < word.size() - 1 && word[i] == min.first &&
                    word[i + 1] == min.second) {
                    newWord.push_back(min.first + min.second);
                    i += 2;
                } else {
                    newWord.push_back(word[i]);
                    i += 1;
                }
            }

            word = newWord;

            if (word.size() == 1)
                break;
        }

        return word;
    }

    std::vector<std::string> word_piece(const std::string &token) {

        if (token.empty())
            return {token};

        std::string unk_token = "[UNK]";
        int max_input_chars_per_word = 200;
        std::vector<std::string> word;

        if (token.length() > max_input_chars_per_word) {
            word.push_back(unk_token);
            return word;
        }

        bool is_bad = false;
        int start = 0;
        while (start < token.length()) {
            int end = token.length();
            std::string cur_substr = "";
            while (start < end) {
                std::string sub_str = token.substr(start, (end - start));
                if (start > 0) {
                    sub_str = "##" + sub_str;
                }
                if (tokenizer_token2idx.find(sub_str) !=
                    tokenizer_token2idx.end()) {
                    cur_substr = sub_str;
                    break;
                }
                end--;
            }
            if (cur_substr == "") {
                is_bad = true;
                break;
            }
            word.push_back(cur_substr);
            start = end;
        }

        if (is_bad) {
            word.push_back(unk_token);
        }

        return word;
    }

  public:
    CLIPTokenizer() = default;

    /*
     * language_mode 0 english, 1 chinese;
     */
    int set_language(int language_mode) {
        is_ch_ = language_mode;
        if (is_ch_) {
            UNK_TOKEN_ID = 0;
            BOS_TOKEN_ID = 101;
            EOS_TOKEN_ID = 102;
            PAD_TOKEN_ID = 0;
        }
        return 0;
    }

    int load(const std::string &path) {
        std::string vocab = path + "/vocab.txt";
        if (is_ch_) {
            vocab = path + "/vocab.txt";
        }
        struct stat buffer;
        if (!(stat(vocab.c_str(), &buffer) == 0)) {
            BMFLITE_LOGE("controlnet", "vocab %s not exist!", vocab.c_str());
            return -1;
        }

        std::ifstream infile;
        infile.open(vocab.data());
        std::string s;
        int idx = 0;
        while (getline(infile, s)) {
            tokenizer_token2idx.insert(pair<string, int>(s, idx));
            tokenizer_idx2token.insert(pair<int, string>(idx, s));
            idx++;
        }
        infile.close();
        if (is_ch_) {
            // when chinese, not use bpe merge.
            return 0;
        }

        std::string merge = path + "/merges.txt";
        if (!(stat(merge.c_str(), &buffer) == 0)) {
            BMFLITE_LOGE("controlnet", "merge %s not exist!", merge.c_str());
            return -1;
        }

        bpeRanks = loadBpeRanks(merge);
        return 0;
    }

    std::vector<int> tokenize_discard(std::string text, size_t max_length = 0,
                                      bool padding = false) {
        std::vector<int32_t> tokens = encode(text);
        tokens.insert(tokens.begin(), BOS_TOKEN_ID);
        if (max_length > 0) {
            if (tokens.size() > max_length - 1) {
                tokens.resize(max_length - 1);
            } else {
                if (padding) {
                    tokens.insert(tokens.end(), max_length - 1 - tokens.size(),
                                  PAD_TOKEN_ID);
                }
            }
        }
        tokens.push_back(EOS_TOKEN_ID);
        return tokens;
    }

    bool isChinese(string str) {
        for (int i = 0; i < str.length(); i++) {
            if (str[i] > 127) {
                return true;
            }
        }
        return false;
    }

    std::vector<int> encode(std::string text) {
        std::string original_text = text;
        std::vector<int32_t> bpe_tokens;
        text = whitespace_clean_and_low_case(text);

        std::regex pat(
            R"(<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[[:alpha:]]+|[[:digit:]]|[^[:space:][:alpha:][:digit:]]+)",
            std::regex::icase);

        std::string str = text;
        std::vector<std::string> token_strs;
        std::smatch match_results;
        std::string::const_iterator start = str.begin();
        std::string::const_iterator end = str.end();

        while (std::regex_search(start, end, match_results, pat)) {
            std::string token = match_results[0];
            std::vector<std::string> tokens;

            if (is_ch_) {
                //                tokens = str_bpe_ch(token);
                ////            中文+英文
                int flag = isChinese(token);
                if (flag) {
                    tokens = str_bpe_ch(token);
                } else {
                    tokens = word_piece(token);
                }
            } else {
                tokens = str_bpe_en(token);
            }

            for (const auto &bpe_token : tokens) {
                if (tokenizer_token2idx.find(bpe_token) !=
                    tokenizer_token2idx.end()) {
                    bpe_tokens.push_back(tokenizer_token2idx[bpe_token]);
                    token_strs.push_back(bpe_token);
                } else {
                    BMFLITE_LOGE("controlnet", "unknown token : %s",
                                 bpe_token.c_str());
                }
            }
            start = match_results[0].second; // 更新搜索的起始位置
        }
        std::stringstream ss;
        ss << "[";
        for (auto token : token_strs) {
            ss << "\"" << token << "\", ";
        }
        ss << "]";
        BMFLITE_LOGD("controlnet", "split prompt \"%s\" to tokens %s",
                     original_text.c_str(), ss.str().c_str());
        return bpe_tokens;
    }

    // Ref:
    // https://github.com/AUTOMATIC1111/stable-diffusion-webui/blob/cad87bf4e3e0b0a759afa94e933527c3123d59bc/modules/prompt_parser.py#L345
    //
    // Parses a string with attention tokens and returns a list of pairs: text
    // and its associated weight. Accepted tokens are:
    //   (abc) - increases attention to abc by a multiplier of 1.1
    //   (abc:3.12) - increases attention to abc by a multiplier of 3.12
    //   [abc] - decreases attention to abc by a multiplier of 1.1
    //   \( - literal character '('
    //   \[ - literal character '['
    //   \) - literal character ')'
    //   \] - literal character ']'
    //   \\ - literal character '\'
    //   anything else - just text
    //
    // >>> parse_prompt_attention('normal text')
    // [['normal text', 1.0]]
    // >>> parse_prompt_attention('an (important) word')
    // [['an ', 1.0], ['important', 1.1], [' word', 1.0]]
    // >>> parse_prompt_attention('(unbalanced')
    // [['unbalanced', 1.1]]
    // >>> parse_prompt_attention('\(literal\]')
    // [['(literal]', 1.0]]
    // >>> parse_prompt_attention('(unnecessary)(parens)')
    // [['unnecessaryparens', 1.1]]
    // >>> parse_prompt_attention('a (((house:1.3)) [on] a (hill:0.5), sun,
    // (((sky))).')
    // [['a ', 1.0],
    //  ['house', 1.5730000000000004],
    //  [' ', 1.1],
    //  ['on', 1.0],
    //  [' a ', 1.1],
    //  ['hill', 0.55],
    //  [', sun, ', 1.1],
    //  ['sky', 1.4641000000000006],
    //  ['.', 1.1]]
    std::vector<std::pair<std::string, float>>
    parse_prompt_attention(const std::string &text) {
        std::vector<std::pair<std::string, float>> res;
        std::vector<int> round_brackets;
        std::vector<int> square_brackets;

        float round_bracket_multiplier = 1.1f;
        float square_bracket_multiplier = 1 / 1.1f;

        std::regex re_attention(
            R"(\\\(|\\\)|\\\[|\\\]|\\\\|\\|\(|\[|:([+-]?[.\d]+)\)|\)|\]|[^\\()\[\]:]+|:)");
        std::regex re_break(R"(\s*\bBREAK\b\s*)");

        auto multiply_range = [&](int start_position, float multiplier) {
            for (int p = start_position; p < res.size(); ++p) {
                res[p].second *= multiplier;
            }
        };

        std::smatch m;
        std::string remaining_text = text;

        while (std::regex_search(remaining_text, m, re_attention)) {
            std::string text = m[0];
            std::string weight = m[1];

            if (text == "(") {
                round_brackets.push_back((int)res.size());
            } else if (text == "[") {
                square_brackets.push_back((int)res.size());
            } else if (!weight.empty()) {
                if (!round_brackets.empty()) {
                    multiply_range(round_brackets.back(), std::stof(weight));
                    round_brackets.pop_back();
                }
            } else if (text == ")" && !round_brackets.empty()) {
                multiply_range(round_brackets.back(), round_bracket_multiplier);
                round_brackets.pop_back();
            } else if (text == "]" && !square_brackets.empty()) {
                multiply_range(square_brackets.back(),
                               square_bracket_multiplier);
                square_brackets.pop_back();
            } else if (text == "\\(") {
                res.push_back({text.substr(1), 1.0f});
            } else {
                res.push_back({text, 1.0f});
            }

            remaining_text = m.suffix();
        }

        for (int pos : round_brackets) {
            multiply_range(pos, round_bracket_multiplier);
        }

        for (int pos : square_brackets) {
            multiply_range(pos, square_bracket_multiplier);
        }

        if (res.empty()) {
            res.push_back({"", 1.0f});
        }

        int i = 0;
        while (i + 1 < res.size()) {
            if (res[i].second == res[i + 1].second) {
                res[i].first += res[i + 1].first;
                res.erase(res.begin() + i + 1);
            } else {
                ++i;
            }
        }

        return res;
    }

    std::pair<std::vector<int>, std::vector<float>>
    tokenize(const string text, size_t max_length = 0, bool padding = false) {
        auto parsed_attention = parse_prompt_attention(text);

        {
            std::stringstream ss;
            ss << "[";
            for (const auto &item : parsed_attention) {
                ss << "['" << item.first << "', " << item.second << "], ";
            }
            ss << "]";
            BMFLITE_LOGD("controlnet", "parse '%s' to %s", text.c_str(),
                         ss.str().c_str());
        }

        std::vector<int> tokens;
        std::vector<float> weights;
        for (const auto &item : parsed_attention) {
            const std::string &curr_text = item.first;
            float curr_weight = item.second;
            std::vector<int> curr_tokens = encode(curr_text);
            tokens.insert(tokens.end(), curr_tokens.begin(), curr_tokens.end());
            weights.insert(weights.end(), curr_tokens.size(), curr_weight);
        }
        tokens.insert(tokens.begin(), BOS_TOKEN_ID);
        weights.insert(weights.begin(), 1.0);

        auto tokens_length = tokens.size();

        if (max_length > 0) {
            if (tokens.size() > max_length - 1) {
                tokens.resize(max_length - 1);
                weights.resize(max_length - 1);
            } else {
                if (padding) {
                    tokens.insert(tokens.end(), max_length - 1 - tokens.size(),
                                  PAD_TOKEN_ID);
                    weights.insert(weights.end(),
                                   max_length - 1 - weights.size(), 1.0);
                }
            }
        }

        if (is_ch_ && tokens_length < max_length - 1) {
            tokens.insert(tokens.begin() + tokens_length, EOS_TOKEN_ID);
            weights.insert(weights.begin() + tokens_length, 1.0);
        } else {
            tokens.push_back(EOS_TOKEN_ID);
            weights.push_back(1.0);
        }

        // for (int i = 0; i < tokens.size(); i++) {
        //     std::cout << tokens[i] << ":" << weights[i] << ", ";
        // }
        // std::cout << std::endl;

        return {tokens, weights};
    }
};