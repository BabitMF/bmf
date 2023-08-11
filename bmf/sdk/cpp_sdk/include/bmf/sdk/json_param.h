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

#ifndef BMF_PARSE_JSON_H
#define BMF_PARSE_JSON_H

#include <nlohmann/json.hpp>
#include <string>
#include <bmf/sdk/common.h>
#include <bmf/sdk/bmf_type_info.h>
#include <bmf/sdk/sdk_interface.h>

BEGIN_BMF_SDK_NS
/** @ingroup CppMdSDK
 */
class BMF_API JsonParam {

  public:
    /** @brief JsonParam struct
     */
    JsonParam() = default;

    /** @overide
     *  @param json_param copy json_param
     */
    JsonParam(const JsonParam &json_param);

    /** @overide
     *  @param opt_str content of json string
     */
    JsonParam(std::string opt_str);

    /** @overide
     *  @param json_value json value
     */
    explicit JsonParam(nlohmann::json json_value);

    template <typename T> JsonParam operator[](T name) {
        return JsonParam(json_value_[name]);
    }

    template <typename T> T to() const { return json_value_.get<T>(); }

    template <typename T, typename U> T get(U name) const {
        return json_value_[name].template get<T>();
    }

    /** @brief set value of json value
     *  @param json_value json value
     */
    void set_value(nlohmann::json &value);

    /** @brief load file of json content
     *  @param file_name file name of json content
     *  @return 0 for success, else failed
     */
    int load(std::string file_name);

    /** @brief store json content to file
     *  @param file_name file name of json content
     *  @return 0 for success, else failed
     */
    int store(std::string file_name);

    /** @brief parse json content string
     *  @param content json string
     *  @return 0 for success, else failed
     */
    int parse(std::string content);

    /** @brief judge the json has key
     *  @param name name of key
     *  @return true for has, false for not have
     */
    bool has_key(std::string name);

    /** @brief erase the key content from json param
     *  @param name name of key
     *  @return 0 for success, else for failed
     */
    int erase(std::string name);

    /** @brief get all content from json param
     *  @param name name of key
     *  @return 0 for success, else for failed
     */
    int get_iterated(std::vector<std::pair<std::string, std::string>> &group);

    /** @brief get json object according to the key name
     *  @param name name of key
     *  @param result result of json object
     *  @return 0 for success, else for failed
     */
    int get_object(std::string name, JsonParam &result);

    /** @brief get json object list according to the key name
     *  @param name name of key
     *  @param result result of json object list
     *  @return 0 for success, else for failed
     */
    int get_object_list(std::string name, std::vector<JsonParam> &result);

    /** @brief get string according to the key name
     *  @param name name of key
     *  @param result result of string
     *  @return 0 for success, else for failed
     */
    int get_string(std::string name, std::string &result);

    /** @brief get string list according to the key name
     *  @param name name of key
     *  @param result result of string list
     *  @return 0 for success, else for failed
     */
    int get_string_list(std::string name, std::vector<std::string> &result);

    /** @brief get int according to the key name
     *  @param name name of key
     *  @param result result of int
     *  @return 0 for success, else for failed
     */
    int get_int(std::string name, int &result);

    /** @brief get long value according to the key name
     *  @param name name of key
     *  @param result result of long
     *  @return 0 for success, else for failed
     */
    int get_long(std::string name, int64_t &result);

    /** @brief get int value list according to the key name
     *  @param name name of key
     *  @param result result of int list
     *  @return 0 for success, else for failed
     */
    int get_int_list(std::string name, std::vector<int> &result);

    /** @brief get double value according to the key name
     *  @param name name of key
     *  @param result result of double
     *  @return 0 for success, else for failed
     */
    int get_double(std::string name, double &result);

    /** @brief get double value list according to the key name
     *  @param name name of key
     *  @param result result of doule list
     *  @return 0 for success, else for failed
     */
    int get_double_list(std::string name, std::vector<double> &result);

    /** @brief dump json object to string
     *  @return json string
     */
    std::string dump() const;

    /** @brief merge json patch to current target
     *  @param json_patch json patch
     */
    void merge_patch(const JsonParam &json_patch);

  public:
    nlohmann::json json_value_;
};

//
template <> struct OpaqueDataInfo<JsonParam> {
    const static int key = OpaqueDataKey::kJsonParam;

    static OpaqueData construct(const JsonParam *json) {
        return std::make_shared<JsonParam>(*json);
    }
};

END_BMF_SDK_NS

//
BMF_DEFINE_TYPE(bmf_sdk::JsonParam)

#endif // BMF_PARSE_JSON_H
