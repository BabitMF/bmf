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
#pragma once

#include "common.h" // BMF_SDK_VERSION

#ifdef __cplusplus
#include <bmf/sdk/video_frame.h>
#include <bmf/sdk/audio_frame.h>
#include <bmf/sdk/bmf_av_packet.h>
#include <bmf/sdk/json_param.h>
#include <bmf/sdk/task.h>
#include <bmf/sdk/packet.h>
#include <bmf/sdk/module.h>
#include <bmf/sdk/module_functor.h>

typedef bmf_sdk::VideoFrame *bmf_VideoFrame;
typedef bmf_sdk::AudioFrame *bmf_AudioFrame;
typedef bmf_sdk::BMFAVPacket *bmf_BMFAVPacket;
typedef bmf_sdk::JsonParam *bmf_JsonParam;
typedef bmf_sdk::Task *bmf_Task;
typedef bmf_sdk::Packet *bmf_Packet;
typedef bmf_sdk::TypeInfo *bmf_TypeInfo;
typedef bmf_sdk::Module *bmf_Module;
typedef bmf_sdk::ModuleTag *bmf_ModuleTag;
typedef bmf_sdk::ModuleInfo *bmf_ModuleInfo;
typedef bmf_sdk::ModuleFunctor *bmf_ModuleFunctor;

extern "C" {

#else //__cplusplus

typedef void *bmf_VideoFrame;
typedef void *bmf_AudioFrame;
typedef void *bmf_BMFAVPacket;
typedef void *bmf_JsonParam;
typedef void *bmf_Task;
typedef void *bmf_Packet;
typedef void *bmf_TypeInfo;
typedef void *bmf_ModuleTag;
typedef void *bmf_ModuleInfo;
typedef void *bmf_ModuleFunctor;

#endif //__cplusplus

#include <hmp_capi.h>

BMF_API void bmf_set_last_error(const char *errstr);
BMF_API const char *bmf_last_error();
BMF_API const char *bmf_sdk_version();

BMF_API char *bmf_strdup(char const *src);
////////// Common ////////////
BMF_API char *bmf_json_param_dump(bmf_JsonParam json);
BMF_API bmf_JsonParam bmf_json_param_parse(const char *str);
BMF_API void bmf_json_param_free(bmf_JsonParam json);

////////// VideoFrame ////////////
BMF_API bmf_VideoFrame bmf_vf_from_frame(hmp_Frame frame);
BMF_API bmf_VideoFrame bmf_vf_make_frame(int width, int height,
                                         const hmp_PixelInfo pix_info,
                                         const char *device);
BMF_API void bmf_vf_free(bmf_VideoFrame vf);

BMF_API bool bmf_vf_defined(const bmf_VideoFrame vf);
BMF_API int bmf_vf_width(const bmf_VideoFrame vf);
BMF_API int bmf_vf_height(const bmf_VideoFrame vf);
BMF_API int bmf_vf_dtype(const bmf_VideoFrame vf);
BMF_API const hmp_Frame bmf_vf_frame(const bmf_VideoFrame vf);

BMF_API int bmf_vf_device_type(const bmf_VideoFrame vf);
BMF_API int bmf_vf_device_index(const bmf_VideoFrame vf);
BMF_API bmf_VideoFrame bmf_vf_cpu(const bmf_VideoFrame vf, bool non_blocking);
BMF_API bmf_VideoFrame bmf_vf_cuda(const bmf_VideoFrame vf);
BMF_API void bmf_vf_copy_from(bmf_VideoFrame vf, const bmf_VideoFrame from);
BMF_API bmf_VideoFrame bmf_vf_to_device(const bmf_VideoFrame vf,
                                        const char *device, bool non_blocking);
BMF_API void bmf_vf_copy_props(bmf_VideoFrame vf, const bmf_VideoFrame from);
BMF_API void bmf_vf_private_merge(bmf_VideoFrame vf, const bmf_VideoFrame from);
BMF_API const bmf_JsonParam
bmf_vf_private_get_json_param(const bmf_VideoFrame vf);
BMF_API void bmf_vf_private_attach_json_param(bmf_VideoFrame vf,
                                              const bmf_JsonParam json_param);

BMF_API bmf_VideoFrame bmf_vf_reformat(const bmf_VideoFrame vf,
                                       const hmp_PixelInfo pix_info);

BMF_API void bmf_vf_set_pts(bmf_VideoFrame vf, int64_t pts);
BMF_API int64_t bmf_vf_pts(bmf_VideoFrame vf);
BMF_API void bmf_vf_set_time_base(bmf_VideoFrame vf, int num, int den);
BMF_API void bmf_vf_time_base(bmf_VideoFrame vf, int *num, int *den);

BMF_API bool bmf_vf_ready(const bmf_VideoFrame vf);
BMF_API void bmf_vf_record(bmf_VideoFrame vf, bool use_current);
BMF_API void bmf_vf_synchronize(bmf_VideoFrame vf);

////////// AudioFrame ////////////
BMF_API bmf_AudioFrame bmf_af_make_from_data(hmp_Tensor *data, int size,
                                             uint64_t layout, bool planer);
BMF_API bmf_AudioFrame bmf_af_make(int samples, uint64_t layout, bool planer,
                                   int dtype);
BMF_API void bmf_af_free(bmf_AudioFrame af);
BMF_API bool bmf_af_defined(const bmf_AudioFrame af);
BMF_API uint64_t bmf_af_layout(const bmf_AudioFrame af);
BMF_API int bmf_af_dtype(const bmf_AudioFrame af);
BMF_API bool bmf_af_planer(const bmf_AudioFrame af);
BMF_API int bmf_af_nsamples(const bmf_AudioFrame af);
BMF_API int bmf_af_nchannels(const bmf_AudioFrame af);
BMF_API void bmf_af_set_sample_rate(const bmf_AudioFrame af, float ar);
BMF_API float bmf_af_sample_rate(const bmf_AudioFrame af);
BMF_API int bmf_af_planes(const bmf_AudioFrame af, hmp_Tensor *data);
BMF_API int bmf_af_nplanes(const bmf_AudioFrame af);
BMF_API hmp_Tensor bmf_af_plane(const bmf_AudioFrame af, int p);
BMF_API void bmf_af_copy_props(bmf_AudioFrame af, const bmf_AudioFrame from);
BMF_API void bmf_af_private_merge(bmf_AudioFrame af, const bmf_AudioFrame from);
BMF_API const bmf_JsonParam
bmf_af_private_get_json_param(const bmf_AudioFrame af);
BMF_API void bmf_af_private_attach_json_param(bmf_AudioFrame af,
                                              const bmf_JsonParam json_param);
BMF_API void bmf_af_set_pts(bmf_AudioFrame af, int64_t pts);
BMF_API int64_t bmf_af_pts(bmf_AudioFrame af);
BMF_API void bmf_af_set_time_base(bmf_AudioFrame af, int num, int den);
BMF_API void bmf_af_time_base(bmf_AudioFrame af, int *num, int *den);

////////// BMFAVPacket ////////////
BMF_API bmf_BMFAVPacket bmf_pkt_make_from_data(hmp_Tensor data);
BMF_API bmf_BMFAVPacket bmf_pkt_make(int size, int dtype);
BMF_API void bmf_pkt_free(bmf_BMFAVPacket pkt);
BMF_API bool bmf_pkt_defined(const bmf_BMFAVPacket pkt);
BMF_API hmp_Tensor bmf_pkt_data(const bmf_BMFAVPacket pkt);
BMF_API void *bmf_pkt_data_ptr(const bmf_BMFAVPacket pkt);
BMF_API const void *bmf_pkt_data_const_ptr(const bmf_BMFAVPacket pkt);
BMF_API int bmf_pkt_nbytes(const bmf_BMFAVPacket pkt);
BMF_API void bmf_pkt_copy_props(bmf_BMFAVPacket pkt,
                                const bmf_BMFAVPacket from);
BMF_API void bmf_pkt_private_merge(bmf_BMFAVPacket pkt,
                                   const bmf_BMFAVPacket from);
BMF_API const bmf_JsonParam
bmf_pkt_private_get_json_param(const bmf_BMFAVPacket pkt);
BMF_API void bmf_pkt_private_attach_json_param(bmf_BMFAVPacket pkt,
                                               const bmf_JsonParam json_param);
BMF_API void bmf_pkt_set_pts(bmf_BMFAVPacket pkt, int64_t pts);
BMF_API int64_t bmf_pkt_pts(bmf_BMFAVPacket pkt);
BMF_API void bmf_pkt_set_time_base(bmf_BMFAVPacket pkt, int num, int den);
BMF_API void bmf_pkt_time_base(bmf_BMFAVPacket pkt, int *num, int *den);
BMF_API int64_t bmf_pkt_offset(bmf_BMFAVPacket pkt);
BMF_API int64_t bmf_pkt_whence(bmf_BMFAVPacket pkt);

/////////// TypeInfo ///////////
BMF_API const char *bmf_type_info_name(const bmf_TypeInfo type_info);
BMF_API unsigned long bmf_type_info_index(const bmf_TypeInfo type_info);

/////////// Packet /////////////
BMF_API void bmf_packet_free(bmf_Packet pkt);
BMF_API int bmf_packet_defined(bmf_Packet pkt);
BMF_API const bmf_TypeInfo bmf_packet_type_info(const bmf_Packet pkt);

BMF_API bmf_Packet bmf_packet_generate_eos_packet();
BMF_API bmf_Packet bmf_packet_generate_eof_packet();
BMF_API bmf_Packet bmf_packet_generate_empty_packet();

BMF_API int64_t bmf_packet_timestamp(const bmf_Packet pkt);
BMF_API void bmf_packet_set_timestamp(bmf_Packet pkt, int64_t timestamp);

BMF_API bmf_Packet bmf_packet_from_videoframe(const bmf_VideoFrame vf);
BMF_API bmf_VideoFrame bmf_packet_get_videoframe(const bmf_Packet pkt);
BMF_API int bmf_packet_is_videoframe(const bmf_Packet pkt);

BMF_API bmf_Packet bmf_packet_from_audioframe(const bmf_AudioFrame af);
BMF_API bmf_AudioFrame bmf_packet_get_audioframe(const bmf_Packet pkt);
BMF_API int bmf_packet_is_audioframe(const bmf_Packet pkt);

BMF_API bmf_Packet
bmf_packet_from_bmfavpacket(const bmf_BMFAVPacket bmf_av_pkt);
BMF_API bmf_BMFAVPacket bmf_packet_get_bmfavpacket(const bmf_Packet pkt);
BMF_API int bmf_packet_is_bmfavpacket(const bmf_Packet pkt);

BMF_API bmf_Packet bmf_packet_from_json_param(const bmf_JsonParam json);
BMF_API bmf_JsonParam bmf_packet_get_json_param(const bmf_Packet pkt);

BMF_API bmf_Packet bmf_packet_from_string_param(char *const str);
BMF_API char *const bmf_packet_get_string_param(const bmf_Packet pkt);
BMF_API int bmf_packet_is_json_param(const bmf_Packet pkt);

////////// Task ////////////
BMF_API bmf_Task bmf_task_make(int node_id, int *istream_ids, int ninputs,
                               int *ostream_ids, int noutputs);
BMF_API void bmf_task_free(bmf_Task task);
BMF_API int bmf_task_fill_input_packet(bmf_Task task, int stream_id,
                                       const bmf_Packet packet);
BMF_API int bmf_task_fill_output_packet(bmf_Task task, int stream_id,
                                        const bmf_Packet packet);
BMF_API bmf_Packet bmf_task_pop_packet_from_out_queue(bmf_Task task,
                                                      int stream_id);
BMF_API bmf_Packet bmf_task_pop_packet_from_input_queue(bmf_Task task,
                                                        int stream_id);
BMF_API int64_t bmf_task_timestamp(const bmf_Task task);
BMF_API void bmf_task_set_timestamp(const bmf_Task task, int64_t timestamp);
BMF_API int bmf_task_get_input_stream_ids(bmf_Task task, int *ids);
BMF_API int bmf_task_get_output_stream_ids(bmf_Task task, int *ids);
BMF_API int bmf_task_get_node(bmf_Task task);

////////// Module ///////////

////////// ModuleTag ///////////
BMF_API bmf_ModuleTag bmf_module_tag_make(int64_t tag);
BMF_API void bmf_module_tag_free(bmf_ModuleTag tag);

////////// ModuleInfo ///////////
BMF_API bmf_ModuleInfo bmf_module_info_make();
BMF_API void bmf_module_info_free(bmf_ModuleInfo);
BMF_API void bmf_module_info_set_description(bmf_ModuleInfo info,
                                             const char *description);
BMF_API void bmf_module_info_set_tag(bmf_ModuleInfo info,
                                     const bmf_ModuleTag tag);

///////// ModuleFunctor /////////////
BMF_API bmf_ModuleFunctor bmf_module_functor_make(
    const char *name, const char *type, const char *path, const char *entry,
    const char *option, int ninputs, int noutputs, int node_id);
BMF_API void bmf_module_functor_free(bmf_ModuleFunctor mf);
BMF_API bmf_Packet *bmf_module_functor_call(bmf_ModuleFunctor mf,
                                            const bmf_Packet *inputs,
                                            int ninputs, int *noutputs,
                                            bool *is_done);
BMF_API int bmf_module_functor_execute(bmf_ModuleFunctor mf,
                                       const bmf_Packet *inputs, int ninputs,
                                       bool cleanup, bool *is_done);
BMF_API bmf_Packet *bmf_module_functor_fetch(bmf_ModuleFunctor mf, int index,
                                             int *noutputs, bool *is_done);

#ifdef __cplusplus
} // extern "C"
#endif
