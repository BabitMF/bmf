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

#if defined(__clang__)
#define BMF_FUNC_VIS __attribute__((__visibility__("default")))
#elif defined(__GNUC__)
#define BMF_FUNC_VIS __attribute__((visibility("default")))
#else
#define BMF_FUNC_VIS
#endif

// From SDK
#include <bmf/sdk/cbytes.h>
#include <bmf/sdk/packet.h>
#include <bmf/sdk/module.h>

#include <memory>
#include <string>
#include <stdint.h>

namespace bmf {
/*
 * @brief Interface of a runnable BMF Graph instance.
 */
class BMFGraph {
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
    BMF_FUNC_VIS BMFGraph(std::string const &graph_config, bool is_path = false,
                          bool need_merge = true);

    BMF_FUNC_VIS BMFGraph(BMFGraph const &graph);

    BMF_FUNC_VIS ~BMFGraph();

    BMF_FUNC_VIS BMFGraph &operator=(BMFGraph const &graph);

    BMF_FUNC_VIS uint32_t uid();

    /*
     * @brief Start running a BMF Graph instance.
     */
    BMF_FUNC_VIS void start();

    /*
     * @brief Update a dynamical BMF Graph instance.
     */
    BMF_FUNC_VIS void update(const std::string &config, bool is_path);

    /*
     * @brief Wait a running BMF Graph instance stopping.
     *      It may be stuck if the instance has not been 'start'ed.
     */
    BMF_FUNC_VIS int close();

    BMF_FUNC_VIS int force_close();

    /*
     * @brief
     * @param stream_name [in]
     * @param packet [in]
     *
     * @return
     */
    BMF_FUNC_VIS int add_input_stream_packet(std::string const &stream_name,
                                             bmf_sdk::Packet &packet,
                                             bool block = false);

    /*
     * @brief
     * @param stream_name [in]
     * @param block [in]
     *
     * @return
     */
    BMF_FUNC_VIS bmf_sdk::Packet
    poll_output_stream_packet(std::string const &stream_name,
                              bool block = true);

    /*
     * @brief
     * @return
     */
    BMF_FUNC_VIS GraphRunningInfo status();

  private:
    uint32_t graph_uid_;
};

/*
 * @brief Interface of a Module instance.
 */
class BMFModule {
  public:
    /*
     * @brief
     * @param [in] module_name
     * @param [in] option
     * @param [in] module_type
     * @param [in] module_path
     * @param [in] module_entry
     */
    BMF_FUNC_VIS BMFModule(std::string const &module_name,
                           std::string const &option,
                           std::string const &module_type = "",
                           std::string const &module_path = "",
                           std::string const &module_entry = "");

    /*
     * @brief
     * @param [in] module_p
     */
    BMF_FUNC_VIS BMFModule(std::shared_ptr<bmf_sdk::Module> module_p);

    BMF_FUNC_VIS BMFModule(BMFModule const &mod);

    BMF_FUNC_VIS ~BMFModule();

    BMF_FUNC_VIS BMFModule &operator=(BMFModule const &mod);

    /*
     * @brief
     * @return
     */
    BMF_FUNC_VIS uint32_t uid();

    /*
     * @brief
     * @return
     */
    BMF_FUNC_VIS int32_t process(bmf_sdk::Task &task);

    /*
     * @brief
     * @return
     */
    BMF_FUNC_VIS int32_t reset();

    /*
     * @brief
     * @return
     */
    BMF_FUNC_VIS int32_t init();

    /*
     * @brief
     * @return
     */
    BMF_FUNC_VIS int32_t close();

  private:
    friend BMFGraph;

    uint32_t module_uid_;
    std::string module_name_;
};

/*
 * @brief Interface of a registered callback.
 */
class BMFCallback {
  public:
    /*
     * @brief
     * @param [in] callback
     */
    BMF_FUNC_VIS
    BMFCallback(std::function<bmf_sdk::CBytes(bmf_sdk::CBytes)> callback);

    BMF_FUNC_VIS BMFCallback(BMFCallback const &cb);

    BMF_FUNC_VIS ~BMFCallback();

    BMF_FUNC_VIS BMFCallback &operator=(BMFCallback const &cb);

    /*
     * @brief
     * @return
     */
    BMF_FUNC_VIS uint32_t uid();

  private:
    friend BMFGraph;

    uint32_t callback_uid_;
};

void ChangeDmpPath(std::string path);
} // namespace bmf

#endif // CONNECTOR_CONNECTOR_HPP
