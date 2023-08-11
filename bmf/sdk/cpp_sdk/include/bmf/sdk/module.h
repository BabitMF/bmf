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

#ifndef BMF_MODULE_H
#define BMF_MODULE_H

#include <functional>
#include <iostream>
#include <stdint.h>
#include <bmf/sdk/config.h>
#include <bmf/sdk/cbytes.h>
#include <bmf/sdk/common.h>
#include <bmf/sdk/json_param.h>
#include <bmf/sdk/task.h>
#include <bmf/sdk/log.h>

namespace bmf_sdk {
/** @ingroup CppMdSDK
 */
class BMF_API Module {
  public:
    /** @brief
        @param node_id unique id .
        @param json_param json param of module.
        */
    Module(int32_t node_id = -1, JsonParam json_param = JsonParam()) {
        configure_bmf_log();
        node_id_ = node_id;
    };

    /** @brief get input stream info of module
        @param json_param input stream info.
        @return 0 for success, else for failed.
        */
    virtual int32_t get_input_stream_info(JsonParam &json_param) { return 0; };

    /** @brief set input stream info of module
        @param json_param input stream info.
        @return 0 for success, else for failed.
        */
    virtual int32_t set_input_stream_info(JsonParam &json_param) { return 0; };

    /** @brief set output stream info of module
        @param json_param output stream info.
        @return 0 for success, else for failed.
        */
    virtual int32_t set_output_stream_info(JsonParam &json_param) { return 0; };

    /** @brief get output stream info of module
        @param json_param output stream info.
        @return 0 for success, else for failed.
        */
    virtual int32_t get_output_stream_info(JsonParam &json_param) { return 0; };

    /** @brief get info of module
        @param json_param module info.
        @return 0 for success, else for failed.
        */
    virtual int32_t get_module_info(JsonParam &json_param) { return 0; };

    /** @brief init module
        @return 0 for success, else for failed.
        */
    virtual int32_t init() { return 0; };

    /** @brief reset module
        @return 0 for success, else for failed.
        */
    virtual int32_t reset() { return 0; };

    /** @brief set module mode to flush data
        @return 0 for success, else for failed.
        */
    virtual int32_t flush() { return 0; };

    /** @brief dynamic reset module according to the jsonParam
     *  @param opt_reset json param of reset
        @return 0 for success, else for failed.
        */
    virtual int32_t dynamic_reset(JsonParam opt_reset) { return 0; };

    /** @brief process task
    *  @param task need to be processed
       @return 0 for success, else for failed.
       */
    virtual int32_t process(Task &task) = 0;

    /** @brief close module and release resources
       @return 0 for success, else for failed.
       */
    virtual int32_t close() { return 0; };

    /** @brief check the input stream if need hungry check
     * @param input_stream_id input stream id
       @return true if the input need to check if hungry, else is false.
       */
    virtual bool need_hungry_check(int input_stream_id) { return false; };

    /** @brief check the input stream if need data
     * @param input_stream_id input stream id
       @return true if the input stream need data.
       */
    virtual bool is_hungry(int input_stream_id) { return true; };

    /** @brief check the module type
       @return true if the module is infinity.
       */
    virtual bool is_infinity() { return false; };

    /** @brief set the graph callback of module
     * @param callback_endpoint callback that defined in graph
     */
    virtual void
    set_callback(std::function<CBytes(int64_t, CBytes)> callback_endpoint){};

    /** @brief check the module is subgraph
     * @return true if the module is subgraph, else is false
     */
    virtual bool is_subgraph() { return false; };

    /** @brief if the module is subgraph get the graph config
     * @param json_param return value of config
     * @return true if the module is subgraph and has the graph config, else is
     * false
     */
    virtual bool get_graph_config(JsonParam &json_param) { return false; }

    /**
     * @brief report module stats
     *
     * @param json_param stats
     * @param hints hints pass to stats caculation
     * @return int32_t
     */
    virtual int32_t report(JsonParam &json_param, int hints = 0) { return 0; };

    virtual ~Module(){};

    int32_t node_id_ = -1;
};

} // namespace bmf_sdk

#endif // BMF_MODULE_H
