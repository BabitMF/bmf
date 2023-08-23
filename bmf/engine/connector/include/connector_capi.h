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

#include <bmf/sdk/bmf_capi.h>

#ifdef __cplusplus

#include "connector.hpp"

//
typedef bmf::BMFGraph *bmf_BMFGraph;
typedef bmf::BMFModule *bmf_BMFModule;

extern "C" {

#else //__cplusplus

typedef void *bmf_BMFGraph;
typedef void *bmf_BMFModule;

#endif //__cplusplus

BMF_API const char *bmf_engine_last_error();

//////////////// bmf::BMFGraph ////////////////
BMF_API bmf_BMFGraph bmf_make_graph(char const *graph_json, bool is_path,
                                    bool need_merge);
BMF_API void bmf_graph_free(bmf_BMFGraph graph);
BMF_API uint32_t bmf_graph_uid(bmf_BMFGraph graph);
BMF_API int bmf_graph_start(bmf_BMFGraph graph);
BMF_API int bmf_graph_close(bmf_BMFGraph graph);
BMF_API int bmf_graph_add_input_stream_packet(bmf_BMFGraph graph,
                                              char const *stream_name,
                                              bmf_Packet packet, bool block);
BMF_API bmf_Packet bmf_graph_poll_output_stream_packet(bmf_BMFGraph graph,
                                                       char const *stream_name);
BMF_API int bmf_graph_update(bmf_BMFGraph graph, char const *config,
                             bool is_path);
BMF_API int bmf_graph_force_close(bmf_BMFGraph graph);
BMF_API char *bmf_graph_status(bmf_BMFGraph graph);

///////////////// bmf::BMFModule ////////////////
BMF_API bmf_BMFModule bmf_make_module(char const *module_name,
                                      char const *option,
                                      char const *module_type,
                                      char const *module_path,
                                      char const *module_entry);
BMF_API void bmf_module_free(bmf_BMFModule module);
BMF_API int bmf_module_uid(bmf_BMFModule module);
BMF_API int bmf_module_process(bmf_BMFModule module, bmf_Task task);
BMF_API int bmf_module_init(bmf_BMFModule module);
BMF_API int bmf_module_reset(bmf_BMFModule module);
BMF_API int bmf_module_close(bmf_BMFModule module);

#ifdef __cplusplus
} // extern "C"
#endif