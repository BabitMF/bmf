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

#include <nlohmann/json.hpp>

#include <bmf/sdk/common.h>
#include <bmf/sdk/json_param.h>

#include <string>
#include <vector>

BEGIN_BMF_ENGINE_NS
USE_BMF_SDK_NS

class ModuleConfig {
  public:
    ModuleConfig() = default;

    ModuleConfig(nlohmann::json &module_config);

    ModuleConfig(JsonParam &module_config);

    ModuleConfig(const ModuleConfig &other);

    std::string get_module_name();
    std::string get_module_type();
    std::string get_module_path();
    std::string get_module_entry();

    nlohmann::json to_json();

    std::string module_name;
    std::string module_type;
    std::string module_path;
    std::string module_entry;

    bool operator==(ModuleConfig const &rhs) {
        return this->module_name == rhs.module_name &&
               this->module_type == rhs.module_type &&
               this->module_path == rhs.module_path &&
               this->module_entry == rhs.module_entry;
    }

  private:
    void init(nlohmann::json &module_config);
};

class StreamConfig {
  public:
    StreamConfig() = default;

    StreamConfig(nlohmann::json &stream_config);

    StreamConfig(JsonParam &stream_config);

    StreamConfig(const StreamConfig &other);

    std::string get_alias();

    void set_identifier(std::string i);

    std::string get_identifier();

    std::string get_notify();

    nlohmann::json to_json();

    std::string identifier;
    std::string alias;
    std::string notify;

    bool operator==(StreamConfig const &rhs) {
        return this->identifier == rhs.identifier && this->alias == rhs.alias &&
               this->notify == rhs.notify;
    }

  private:
    void init(nlohmann::json &stream_config);
};

class NodeMetaInfo {
  public:
    NodeMetaInfo() = default;

    NodeMetaInfo(JsonParam &node_meta);

    NodeMetaInfo(nlohmann::json &node_meta);

    NodeMetaInfo(const NodeMetaInfo &other);

    int32_t get_premodule_id();

    int32_t get_bundle();

    uint32_t get_queue_size_limit();

    std::map<int64_t, uint32_t> get_callback_binding();

    nlohmann::json to_json();

    bool operator==(NodeMetaInfo const &rhs) {
        return this->premodule_id == rhs.premodule_id &&
               this->bundle == rhs.bundle &&
               this->queue_size_limit == rhs.queue_size_limit;
    }

  private:
    void init(nlohmann::json &node_meta);

    int32_t premodule_id = -1;
    int32_t bundle = -1;
    uint32_t queue_size_limit = 5;
    std::map<int64_t, uint32_t> callback_binding;
};

class NodeConfig {
  public:
    NodeConfig() = default;

    NodeConfig(nlohmann::json &node_config);

    NodeConfig(JsonParam &node_config);

    NodeConfig(const NodeConfig &other);

    // NodeConfig(NodeConfig &&other) noexcept;

    ModuleConfig get_module_info();

    NodeMetaInfo get_node_meta();

    JsonParam get_option();

    void set_option(JsonParam node_option);

    std::vector<StreamConfig> &get_input_streams();

    std::vector<StreamConfig> &get_output_streams();

    void add_input_stream(StreamConfig input_stream);

    void add_output_stream(StreamConfig output_stream);

    void change_input_stream_identifier(std::string identifier);

    void change_output_stream_identifier(size_t order = 0);

    std::string get_input_manager();

    void set_output_manager(std::string output_manager_type);

    std::string get_output_manager();

    void set_id(int id);

    int get_id();

    void set_scheduler(int scheduler);

    int get_scheduler();

    void set_thread(int thread);

    int get_thread();

    std::string get_alias();

    std::string get_action();

    nlohmann::json to_json();

    // // Assignment operator
    // NodeConfig& operator=(const NodeConfig &other) {
    //     if (this != &other) {
    //         id = other.id;
    //         module = other.module;
    //         meta = other.meta;
    //         input_streams = other.input_streams;  // Copying vectors
    //         output_streams = other.output_streams;
    //         option = other.option;
    //         scheduler = other.scheduler;
    //         thread = other.thread;
    //         input_manager = other.input_manager;
    //         output_manager = other.output_manager;
    //         alias = other.alias;
    //         action = other.action;
    //     }
    //     return *this;
    // }

    // // Move assignment operator
    // NodeConfig& operator=(NodeConfig &&other) noexcept {
    //     if (this != &other) {
    //         id = std::move(other.id);
    //         module = std::move(other.module);
    //         meta = std::move(other.meta);
    //         input_streams = std::move(other.input_streams);  // Move the vectors
    //         output_streams = std::move(other.output_streams);
    //         option = std::move(other.option);
    //         scheduler = std::move(other.scheduler);
    //         thread = std::move(other.thread);
    //         input_manager = std::move(other.input_manager);
    //         output_manager = std::move(other.output_manager);
    //         alias = std::move(other.alias);
    //         action = std::move(other.action);
    //     }
    //     return *this;
    // }

    bool operator==(NodeConfig const &rhs) {
        return this->id == rhs.id && this->module == rhs.module &&
               this->meta == rhs.meta;
    }

    int id;
    ModuleConfig module;
    NodeMetaInfo meta;
    std::vector<StreamConfig> input_streams;
    std::vector<StreamConfig> output_streams;
    JsonParam option;
    int scheduler;
    int thread = 1;
    std::string input_manager = "immediate";
    std::string output_manager = "default";
    std::string alias;
    std::string action;

  private:
    void init(nlohmann::json &node_config);
};

class GraphConfig {
  public:
    GraphConfig() = default;

    GraphConfig(std::string config_file);

    GraphConfig(nlohmann::json &graph_config);

    GraphConfig(JsonParam &graph_json);

    JsonParam get_option();

    BmfMode get_mode();

    std::vector<NodeConfig> get_nodes();

    std::vector<StreamConfig> get_input_streams();

    std::vector<StreamConfig> get_output_streams();

    nlohmann::json to_json();

    std::vector<NodeConfig> nodes;
    BmfMode mode;
    std::vector<StreamConfig> input_streams;
    std::vector<StreamConfig> output_streams;
    JsonParam option;

  private:
    void init(nlohmann::json &graph_config);
};
END_BMF_ENGINE_NS

#endif // BMF_GRAPH_CONFIG_H
