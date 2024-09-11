/*
 * Copyright 2024 Babit Authors
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

#ifndef _BMFLITE_PARAM_H_
#define _BMFLITE_PARAM_H_

#include "bmf_common.h"
#include <string>

namespace bmf_lite {

/** @ingroup CppMdSDK
 */
class ParamImpl;
class BMF_LITE_EXPORT Param {
  public:
    /** @brief Param struct
     */
    Param();

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

    /** @brief get string according to the key name
     *  @param name name of key
     *  @param result result of string
     *  @return 0 for success, else for failed
     */
    int getString(std::string name, std::string &result);
    int setString(std::string name, std::string result);

    /** @brief get string list according to the key name
     *  @param name name of key
     *  @param result result of string list
     *  @return 0 for success, else for failed
     */
    int getStringList(std::string name, std::vector<std::string> &result);
    int setStringList(std::string name, std::vector<std::string> result);

    /** @brief get int according to the key name
     *  @param name name of key
     *  @param result result of int
     *  @return 0 for success, else for failed
     */
    int getInt(std::string name, int &result);
    int setInt(std::string name, int result);

    /** @brief get long value according to the key name
     *  @param name name of key
     *  @param result result of long
     *  @return 0 for success, else for failed
     */
    int getLong(std::string name, int64_t &result);
    int setLong(std::string name, int64_t result);
    /** @brief get int value list according to the key name
     *  @param name name of key
     *  @param result result of int list
     *  @return 0 for success, else for failed
     */
    int getIntList(std::string name, std::vector<int> &result);
    int setIntList(std::string name, std::vector<int> result);
    /** @brief get double value according to the key name
     *  @param name name of key
     *  @param result result of double
     *  @return 0 for success, else for failed
     */
    int getDouble(std::string name, double &result);
    int setDouble(std::string name, double result);
    /** @brief get double value list according to the key name
     *  @param name name of key
     *  @param result result of doule list
     *  @return 0 for success, else for failed
     */
    int getDoubleList(std::string name, std::vector<double> &result);
    int setDoubleList(std::string name, std::vector<double> result);

    /** @brief get float value according to the key name
     *  @param name name of key
     *  @param result result of float
     *  @return 0 for success, else for failed
     */
    int getFloat(std::string name, float &result);
    int setFloat(std::string name, float result);
    /** @brief get float value list according to the key name
     *  @param name name of key
     *  @param result result of float list
     *  @return 0 for success, else for failed
     */
    int getFloatList(std::string name, std::vector<float> &result);
    int setFloatList(std::string name, std::vector<float> result);
    /** @brief dump json object to string
     *  @return json string
     */
    std::string dump();

  public:
    std::shared_ptr<ParamImpl> param_impl_;
};

} // namespace bmf_lite

#endif // _BMFLITE_PARAM_H_