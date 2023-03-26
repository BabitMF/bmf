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

#include <gtest/gtest.h>

USE_BMF_SDK_NS

TEST(json_param, parse_json) {
     JsonParam json_param;
     std::string path = getcwd(NULL,0);
     std::string file_name = "../../example/run_by_config/config.json";
     json_param.load(file_name);
     std::string mode;
     json_param.get_string("mode", mode);
     EXPECT_EQ(mode, "normal");
     std::vector<JsonParam> node_config_list;
     json_param.get_object_list("nodes", node_config_list);
     JsonParam json_param2;
     json_param2 = json_param;
     EXPECT_EQ(node_config_list.size(), 4);
     JsonParam module_info;
     std::string module_name;
     node_config_list[0].get_object("module_info", module_info);
     module_info.get_string("name", module_name);
     EXPECT_EQ(module_name, "c_ffmpeg_decoder");
}

TEST(json_param, basic_type){
    std::string str = "{\"double\":0.01,\"int\":100,\"long\":999999999}";
    JsonParam json_param = JsonParam(str);

    int i;
    json_param.get_int("int", i);
    EXPECT_EQ(i, 100);

    double d;
    json_param.get_double("double", d);
    EXPECT_EQ(d, 0.01);

    int64_t l;
    json_param.get_long("long", l);
    EXPECT_EQ(l, 999999999);
    std::string value = json_param.dump();
    EXPECT_EQ(value, str);
}

TEST(json_param, basic_list_type){
    std::string str = "{\"int_list\":[1,2,3,4,5,6],\"double_list\":[0.001,0.002,0.003],\"string_list\":[\"test_str_001\",\"test_str_002\"]}";
    bmf_nlohmann::json json_value = bmf_nlohmann::json::parse(str);
    JsonParam json_param = JsonParam(json_value);

    std::vector<int> int_list;
    json_param.get_int_list("int_list", int_list);
    EXPECT_EQ(int_list.size(), 6);

    std::vector<double> double_list;
    json_param.get_double_list("double_list", double_list);
    EXPECT_EQ(double_list.size(), 3);

    std::vector<std::string> string_list;
    json_param.get_string_list("string_list", string_list);
    EXPECT_EQ(string_list.size(), 2);
}

TEST(json_param, remove_element){
    std::string str = "{\"id1\":\"100\",\"id2\":\"200\", \"id3\":\"300\"}";
    JsonParam json_param;
    json_param.parse(str);

    std::vector<std::pair<std::string, std::string>> group;
    json_param.get_iterated(group);
    EXPECT_EQ(group.size(), 3);
    for (auto &it: group){
        if(it.first == "id1"){
            EXPECT_EQ(it.second, "100");
        }else if(it.first == "id2"){
            EXPECT_EQ(it.second, "200");
        }else{
            EXPECT_EQ(it.first, "id3");
            EXPECT_EQ(it.second, "300");
        }
    }

    json_param.erase("id1");
    group.clear();
    json_param.get_iterated(group);
    EXPECT_EQ(group.size(), 2);
    for (auto &it: group){
        if (it.first == "id2"){
            EXPECT_EQ(it.second, "200");
        }else if (it.first == "id3"){
            EXPECT_EQ(it.second, "300");
        }
    }
}