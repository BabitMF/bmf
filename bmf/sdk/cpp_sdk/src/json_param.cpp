/*
 * Copyright 2023 Babit Authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <bmf/sdk/json_param.h>

#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <typeinfo>


BEGIN_BMF_SDK_NS
    JsonParam::JsonParam(const JsonParam &json_param) {
        json_value_ = json_param.json_value_;
    }

    JsonParam::JsonParam(bmf_nlohmann::json json_value) {
        json_value_ = json_value;
    }

    JsonParam::JsonParam(std::string opt_str) {
        json_value_ = bmf_nlohmann::json::parse(opt_str);
    }

    int JsonParam::load(std::string file_name) {
        std::ifstream t(file_name);
        t >> json_value_;
        return 0;
    }

    int JsonParam::parse(std::string content) {
        json_value_ = bmf_nlohmann::json::parse(content);
        return 0;
    }

    bool JsonParam::has_key(std::string name) {
        if (json_value_.empty()) {
            return false;
        }
        if (json_value_.count(name) > 0) {
            return true;
        } else {
            return false;
        }
    }

    int JsonParam::erase(std::string name) {
        if (json_value_.empty()) {
            return 0;
        }
        return json_value_.erase(name);
    }

    int JsonParam::get_iterated(std::vector<std::pair<std::string, std::string>> &group) {
        for (const auto &item : json_value_.items()) {
            std::string name = item.key();
            std::string val = item.value().is_string() ? item.value().get<std::string>() : item.value().dump();
            group.emplace_back(name, val);
        }
        return 0;
    }

    void JsonParam::set_value(bmf_nlohmann::json &value) {
        json_value_ = value;
    }

    int JsonParam::get_object(std::string name, JsonParam &result) {
        if (has_key(name)) {
            bmf_nlohmann::json value = json_value_[name];
            result.set_value(value);
            return 0;
        }
        return -1;
    }

    int JsonParam::get_object_list(std::string name, std::vector<JsonParam> &result) {
        if (has_key(name)) {
            for (auto v:json_value_[name]) {
                JsonParam temp_json_param;
                temp_json_param.set_value(v);
                result.push_back(temp_json_param);
            }
            return 0;
        }
        return -1;
    }

    int JsonParam::get_int(std::string name, int &result) {
        result = json_value_[name].get<int>();
        return 0;
    }

    int JsonParam::get_long(std::string name, int64_t &result) {
        result = json_value_[name].get<int64_t>();
        return 0;
    }

    int JsonParam::get_double(std::string name, double &result) {
        result = json_value_[name].get<double>();
        return 0;
    }

    int JsonParam::get_string(std::string name, std::string &result) {
        result = json_value_[name].get<std::string>();
        return 0;
    }

    int JsonParam::get_int_list(std::string name, std::vector<int> &result) {
        for (auto v:json_value_[name])
            result.push_back(v.get<int>());
        return 0;
    }

    int JsonParam::get_double_list(std::string name, std::vector<double> &result) {
        for (auto v:json_value_[name])
            result.push_back(v.get<double>());
        return 0;
    }

    int JsonParam::get_string_list(std::string name, std::vector<std::string> &result) {
        for (auto v:json_value_[name])
            result.push_back(v.get<std::string>());
        return 0;
    }

    std::string JsonParam::dump() const {
        std::string result = json_value_.dump();
        return result;
    }

END_BMF_SDK_NS
