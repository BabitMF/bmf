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

#ifndef CONNECTOR_CONNECTOR_HPP
#define CONNECTOR_CONNECTOR_HPP

#include "running_info.h"

// From SDK
#include <bmf/sdk/cbytes.h>
#include <bmf/sdk/packet.h>
#include <bmf/sdk/module.h>

#include <memory>
#include <string>
#include <stdint.h>

#include "connector_common.h"

namespace bmf {
/*
 * @brief Interface of a runnable BMF Graph instance.
 */
class BMF_ENGINE_API BMFGraph {
  public:
    /*
     * @brief Create a BMF Graph instance from provided graph_config.
     * @param [in] graph_config Config string in serialized json style.
     * @param [in] is_path When set to true, 'graph_config' means a path to json
     * file containing graph config.
     *      This param is set to False by default.
     * @param [in] need_merge When set to true, BMF will preprocess graph to
     * merge ffmpeg_filter nodes into groups if possible.
     *      A merge operation can make graph running faster, but may also
     * introduce some stability issue.
     *      This param is set to true by default.
     */
    BMFGraph(std::string const &graph_config, bool is_path = false,
                          bool need_merge = true);

    BMFGraph(BMFGraph const &graph);

    ~BMFGraph();

    BMFGraph &operator=(BMFGraph const &graph);

    uint32_t uid();

    /*
     * @brief Start running a BMF Graph instance.
     */
    void start();

    /*
     * @brief Update a dynamical BMF Graph instance.
     */
    void update(const std::string &config, bool is_path);

    /*
     * @brief Wait a running BMF Graph instance stopping.
     *      It may be stuck if the instance has not been 'start'ed.
     */
    int close();

    int force_close();

    /*
     * @brief
     * @param stream_name [in]
     * @param packet [in]
     *
     * @return
     */
    int add_input_stream_packet(std::string const &stream_name,
                                             bmf_sdk::Packet &packet,
                                             bool block = false);

    /*
     * @brief
     * @param stream_name [in]
     * @param block [in]
     *
     * @return
     */
    bmf_sdk::Packet
    poll_output_stream_packet(std::string const &stream_name,
                              bool block = true);

    /*
     * @brief
     * @return
     */
    GraphRunningInfo status();

  private:
    uint32_t graph_uid_;
};

/*
 * @brief Interface of a Module instance.
 */
class BMF_ENGINE_API BMFModule {
  public:
    /*
     * @brief
     * @param [in] module_name
     * @param [in] option
     * @param [in] module_type
     * @param [in] module_path
     * @param [in] module_entry
     */
    BMFModule(std::string const &module_name,
                           std::string const &option,
                           std::string const &module_type = "",
                           std::string const &module_path = "",
                           std::string const &module_entry = "");

    /*
     * @brief
     * @param [in] module_p
     */
    BMFModule(std::shared_ptr<bmf_sdk::Module> module_p);

    BMFModule(BMFModule const &mod);

    ~BMFModule();

    BMFModule &operator=(BMFModule const &mod);

    /*
     * @brief
     * @return
     */
    uint32_t uid();

    /*
     * @brief
     * @return
     */
    int32_t process(bmf_sdk::Task &task);

    /*
     * @brief
     * @return
     */
    int32_t reset();

    /*
     * @brief
     * @return
     */
    int32_t init();

    /*
     * @brief
     * @return
     */
    int32_t close();

  private:
    friend BMFGraph;

    uint32_t module_uid_;
    std::string module_name_;
};

/*
 * @brief Interface of a registered callback.
 */
class BMF_ENGINE_API BMFCallback {
  public:
    /*
     * @brief
     * @param [in] callback
     */
    BMFCallback(std::function<bmf_sdk::CBytes(bmf_sdk::CBytes)> callback);

    BMFCallback(BMFCallback const &cb);

    ~BMFCallback();

    BMFCallback &operator=(BMFCallback const &cb);

    /*
     * @brief
     * @return
     */
    uint32_t uid();

  private:
    friend BMFGraph;

    uint32_t callback_uid_;
};

void BMF_ENGINE_API ChangeDmpPath(std::string path);

JsonParam BMF_ENGINE_API ConvertFilterPara(JsonParam config);
} // namespace bmf

#endif // CONNECTOR_CONNECTOR_HPP
