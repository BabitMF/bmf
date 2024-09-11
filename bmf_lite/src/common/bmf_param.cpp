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

#include "common/bmf_param.h"
#include "common/error_code.h"
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace bmf_lite {
enum class ParamDataType {
    UNKNOWN = 0,
    INT_TYPE = 1,
    LONG_TYPE = 2,
    FLOAT_TYPE = 3,
    DOUBLE_TYPE = 4,
    STRING_TYPE = 5,
    INT_LIST_TYPE = 6,
    FLOAT_LIST_TYPE = 7,
    DOUBLE_LIST_TYPE = 8,
    STRING_LIST_TYPE = 9,
};

struct ParamData {
    ParamDataType data_type = ParamDataType::UNKNOWN;
    int int_data;
    int64_t long_data;
    float float_data;
    double double_data;
    std::string string_data;
    std::vector<int> int_list_data;
    std::vector<float> float_list_data;
    std::vector<double> double_list_data;
    std::vector<std::string> string_list_data;
};

class ParamImpl {
  public:
    std::map<std::string, ParamData> json_value_;
};

Param::Param() { param_impl_ = std::make_shared<ParamImpl>(); }

bool Param::has_key(std::string name) {
    if (param_impl_->json_value_.empty()) {
        return false;
    }
    if (param_impl_->json_value_.count(name) > 0) {
        return true;
    } else {
        return false;
    }
}

int Param::erase(std::string name) {
    if (param_impl_->json_value_.empty() ||
        param_impl_->json_value_.count(name) > 0) {
        return BMF_LITE_StsOk;
    }
    return param_impl_->json_value_.erase(name);
}

int Param::getInt(std::string name, int &result) {
    if (has_key(name) &&
        (param_impl_->json_value_[name].data_type == ParamDataType::INT_TYPE)) {
        result = param_impl_->json_value_[name].int_data;
        return BMF_LITE_StsOk;
    } else {
        return BMF_LITE_StsBadArg;
    }
}

int Param::setInt(std::string name, int result) {
    ParamData param_data;
    param_data.data_type = ParamDataType::INT_TYPE;
    param_data.int_data = result;
    param_impl_->json_value_[name] = param_data;
    return BMF_LITE_StsOk;
}

int Param::getLong(std::string name, int64_t &result) {
    if (has_key(name) && (param_impl_->json_value_[name].data_type ==
                          ParamDataType::LONG_TYPE)) {
        result = param_impl_->json_value_[name].long_data;
        return BMF_LITE_StsOk;
    } else {
        return BMF_LITE_StsBadArg;
    }
    return BMF_LITE_StsOk;
}

int Param::setLong(std::string name, int64_t result) {
    ParamData param_data;
    param_data.data_type = ParamDataType::LONG_TYPE;
    param_data.long_data = result;
    param_impl_->json_value_[name] = param_data;
    return BMF_LITE_StsOk;
}

int Param::getFloat(std::string name, float &result) {
    if (has_key(name) && (param_impl_->json_value_[name].data_type ==
                          ParamDataType::FLOAT_TYPE)) {
        result = param_impl_->json_value_[name].float_data;
        return BMF_LITE_StsOk;
    } else {
        return BMF_LITE_StsBadArg;
    }
    return BMF_LITE_StsOk;
}

int Param::setFloat(std::string name, float result) {
    ParamData param_data;
    param_data.data_type = ParamDataType::FLOAT_TYPE;
    param_data.float_data = result;
    param_impl_->json_value_[name] = param_data;
    return BMF_LITE_StsOk;
}

int Param::getDouble(std::string name, double &result) {
    if (has_key(name) && (param_impl_->json_value_[name].data_type ==
                          ParamDataType::DOUBLE_TYPE)) {
        result = param_impl_->json_value_[name].double_data;
        return BMF_LITE_StsOk;
    } else {
        return BMF_LITE_StsBadArg;
    }
    return BMF_LITE_StsOk;
}

int Param::setDouble(std::string name, double result) {
    ParamData param_data;
    param_data.data_type = ParamDataType::DOUBLE_TYPE;
    param_data.double_data = result;
    param_impl_->json_value_[name] = param_data;
    return BMF_LITE_StsOk;
}

int Param::getString(std::string name, std::string &result) {
    if (has_key(name) && (param_impl_->json_value_[name].data_type ==
                          ParamDataType::STRING_TYPE)) {
        result = param_impl_->json_value_[name].string_data;
        return BMF_LITE_StsOk;
    } else {
        return BMF_LITE_StsBadArg;
    }
    return BMF_LITE_StsOk;
}

int Param::setString(std::string name, std::string result) {
    ParamData param_data;
    param_data.data_type = ParamDataType::STRING_TYPE;
    param_data.string_data = result;
    param_impl_->json_value_[name] = param_data;
    return BMF_LITE_StsOk;
}

int Param::getIntList(std::string name, std::vector<int> &result) {
    if (has_key(name) && (param_impl_->json_value_[name].data_type ==
                          ParamDataType::INT_LIST_TYPE)) {
        result = param_impl_->json_value_[name].int_list_data;
        return 0;
    } else {
        return BMF_LITE_StsBadArg;
    }
    return 0;
}

int Param::setIntList(std::string name, std::vector<int> result) {
    ParamData param_data;
    param_data.data_type = ParamDataType::INT_LIST_TYPE;
    param_data.int_list_data = result;
    param_impl_->json_value_[name] = param_data;
    return 0;
}

int Param::getFloatList(std::string name, std::vector<float> &result) {
    if (has_key(name) && (param_impl_->json_value_[name].data_type ==
                          ParamDataType::FLOAT_LIST_TYPE)) {
        result = param_impl_->json_value_[name].float_list_data;
        return 0;
    } else {
        return BMF_LITE_StsBadArg;
    }
    return 0;
}

int Param::setFloatList(std::string name, std::vector<float> result) {
    ParamData param_data;
    param_data.data_type = ParamDataType::FLOAT_LIST_TYPE;
    param_data.float_list_data = result;
    param_impl_->json_value_[name] = param_data;
    return 0;
}

int Param::getDoubleList(std::string name, std::vector<double> &result) {
    if (has_key(name) && (param_impl_->json_value_[name].data_type ==
                          ParamDataType::DOUBLE_LIST_TYPE)) {
        result = param_impl_->json_value_[name].double_list_data;
        return 0;
    } else {
        return BMF_LITE_StsBadArg;
    }
    return 0;
}

int Param::setDoubleList(std::string name, std::vector<double> result) {
    ParamData param_data;
    param_data.data_type = ParamDataType::DOUBLE_LIST_TYPE;
    param_data.double_list_data = result;
    param_impl_->json_value_[name] = param_data;
    return 0;
}

int Param::getStringList(std::string name, std::vector<std::string> &result) {
    if (has_key(name) && (param_impl_->json_value_[name].data_type ==
                          ParamDataType::STRING_LIST_TYPE)) {
        result = param_impl_->json_value_[name].string_list_data;
        return 0;
    } else {
        return BMF_LITE_StsBadArg;
    }
    return 0;
}

int Param::setStringList(std::string name, std::vector<std::string> result) {
    ParamData param_data;
    param_data.data_type = ParamDataType::STRING_LIST_TYPE;
    param_data.string_list_data = result;
    param_impl_->json_value_[name] = param_data;
    return 0;
}

std::string Param::dump() { return ""; }
} // namespace bmf_lite
