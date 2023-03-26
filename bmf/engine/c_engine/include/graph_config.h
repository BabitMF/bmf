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

#ifndef BMF_GRAPH_CONFIG_H
#define BMF_GRAPH_CONFIG_H

#include <bmf_nlohmann/json.hpp>

#include <bmf/sdk/common.h>
#include <bmf/sdk/json_param.h>

#include <string>
#include <vector>

BEGIN_BMF_ENGINE_NS
    USE_BMF_SDK_NS

    class ModuleConfig {
    public:
        ModuleConfig() = default;

        ModuleConfig(bmf_nlohmann::json &module_config);

        ModuleConfig(JsonParam &module_config);

        std::string get_module_name();
        std::string get_module_type();
        std::string get_module_path();
        std::string get_module_entry();

        bmf_nlohmann::json to_json();

        std::string module_name;
        std::string module_type;
        std::string module_path;
        std::string module_entry;

        bool operator==(ModuleConfig const &rhs){
           return this->module_name==rhs.module_name && this->module_type==rhs.module_type &&
                this->module_path==rhs.module_path && this->module_entry==rhs.module_entry;
        }


    private:
        void init(bmf_nlohmann::json &module_config);
    };

    class StreamConfig {
    public:
        StreamConfig() = default;

        StreamConfig(bmf_nlohmann::json &stream_config);

        StreamConfig(JsonParam &stream_config);

        std::string get_alias();

        std::string get_identifier();

        std::string get_notify();

        bmf_nlohmann::json to_json();

        std::string identifier;
        std::string alias;
        std::string notify;

        bool operator==(StreamConfig const &rhs){
           return this->identifier==rhs.identifier && this->alias==rhs.alias && this->notify==rhs.notify;
        }

    private:
        void init(bmf_nlohmann::json &stream_config);
    };

    class NodeMetaInfo {
    public:
        NodeMetaInfo() = default;

        NodeMetaInfo(JsonParam &node_meta);

        NodeMetaInfo(bmf_nlohmann::json &node_meta);

        int32_t get_premodule_id();

        int32_t get_bundle();

        uint32_t get_queue_size_limit();

        std::map<int64_t, uint32_t> get_callback_binding();

        bmf_nlohmann::json to_json();

        bool operator==(NodeMetaInfo const &rhs){
           return this->premodule_id==rhs.premodule_id && this->bundle==rhs.bundle &&
                this->queue_size_limit==rhs.queue_size_limit;
        }

    private:
        void init(bmf_nlohmann::json &node_meta);

        int32_t premodule_id = -1;
        int32_t bundle = -1;
        uint32_t queue_size_limit = 5;
        std::map<int64_t, uint32_t> callback_binding;
    };

    class NodeConfig {
    public:
        NodeConfig() = default;

        NodeConfig(bmf_nlohmann::json &node_config);

        NodeConfig(JsonParam &node_config);

        ModuleConfig get_module_info();

        NodeMetaInfo get_node_meta();

        JsonParam get_option();

        void set_option(JsonParam node_option);

        std::vector<StreamConfig>& get_input_streams();

        std::vector<StreamConfig>& get_output_streams();

        void add_input_stream(StreamConfig input_stream);

        void add_output_stream(StreamConfig output_stream);

        std::string get_input_manager();

        int get_id();

        int get_scheduler();

        std::string get_alias();

        std::string get_action();

        bmf_nlohmann::json to_json();

        bool operator==(NodeConfig const &rhs){
           return this->id==rhs.id && this->module==rhs.module && this->meta==rhs.meta;
        }

        int id;
        ModuleConfig module;
        NodeMetaInfo meta;
        std::vector<StreamConfig> input_streams;
        std::vector<StreamConfig> output_streams;
        JsonParam option;
        int scheduler;
        std::string input_manager = "immediate";
        std::string alias;
        std::string action;
    private:
        void init(bmf_nlohmann::json &node_config);
    };

    class GraphConfig {
    public:
        GraphConfig() = default;

        GraphConfig(std::string config_file);

        GraphConfig(bmf_nlohmann::json &graph_config);

        GraphConfig(JsonParam &graph_json);

        JsonParam get_option();

        BmfMode get_mode();

        std::vector<NodeConfig> get_nodes();

        std::vector<StreamConfig> get_input_streams();

        std::vector<StreamConfig> get_output_streams();

        bmf_nlohmann::json to_json();

        std::vector<NodeConfig> nodes;
        BmfMode mode;
        std::vector<StreamConfig> input_streams;
        std::vector<StreamConfig> output_streams;
        JsonParam option;

    private:
        void init(bmf_nlohmann::json &graph_config);
    };
END_BMF_ENGINE_NS

#endif //BMF_GRAPH_CONFIG_H
