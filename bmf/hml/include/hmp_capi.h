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

#ifdef __cplusplus
#include <hmp/core/stream.h>
#include <hmp/tensor.h>
#include <hmp/imgproc.h>

extern "C" {

/// core data structures
typedef hmp::Tensor *hmp_Tensor;
typedef hmp::Device *hmp_Device;
typedef hmp::Stream *hmp_Stream;
typedef hmp::StreamGuard *hmp_StreamGuard;
typedef hmp::Scalar *hmp_Scalar;

/// imgproc data structures
typedef hmp::ColorModel *hmp_ColorModel;
typedef hmp::PixelInfo *hmp_PixelInfo;
typedef hmp::Frame *hmp_Frame;

#else //__cplusplus

#include <stddef.h>
#include <stdbool.h>

typedef void *hmp_Tensor;
typedef void *hmp_Device;
typedef void *hmp_Scalar;
typedef void *hmp_Stream;
typedef void *hmp_StreamGuard;

typedef void *hmp_ColorModel;
typedef void *hmp_PixelInfo;
typedef void *hmp_Frame;

#endif //__cplusplus

#ifndef HMP_API
#define HMP_API
#endif

HMP_API const char *hmp_last_error();

/////// hmp_Scalar ///////////
HMP_API hmp_Scalar hmp_scalar_float(double v);
HMP_API hmp_Scalar hmp_scalar_int(int64_t v);
HMP_API hmp_Scalar hmp_scalar_bool(bool v);
HMP_API void hmp_scalar_free(hmp_Scalar scalar);

////////// hmp_Device //////////
HMP_API int hmp_device_count(int device_type);

////////// hmp_Stream /////////
HMP_API hmp_Stream hmp_stream_create(int device_type, uint64_t flags);
HMP_API void hmp_stream_free(hmp_Stream stream);
HMP_API bool hmp_stream_query(hmp_Stream stream);
HMP_API void hmp_stream_synchronize(hmp_Stream stream);
HMP_API uint64_t hmp_stream_handle(const hmp_Stream stream);
HMP_API int hmp_stream_device_type(const hmp_Stream stream);
HMP_API int hmp_stream_device_index(const hmp_Stream stream);
HMP_API void hmp_stream_set_current(const hmp_Stream stream);
HMP_API hmp_Stream hmp_stream_current(int device_type);

HMP_API hmp_StreamGuard hmp_stream_guard_create(hmp_Stream stream);
HMP_API void hmp_stream_guard_free(hmp_StreamGuard guard);

/////// hmp_Tensor ///////////
HMP_API hmp_Tensor hmp_tensor_empty(const int64_t *shape, int ndim, int type,
                                    const char *device, bool pinned_memory);
HMP_API hmp_Tensor hmp_tensor_arange(int64_t start, int64_t end, int64_t step,
                                     int type, const char *device,
                                     bool pinned_memory);
HMP_API void hmp_tensor_free(hmp_Tensor tensor);
HMP_API const char *hmp_tensor_stringfy(hmp_Tensor tensor, int *size);

HMP_API void hmp_tensor_fill(hmp_Tensor tensor, hmp_Scalar value);

HMP_API bool hmp_tensor_defined(const hmp_Tensor tensor);
HMP_API int64_t hmp_tensor_dim(const hmp_Tensor tensor);
HMP_API int64_t hmp_tensor_size(const hmp_Tensor tensor, int64_t dim);
HMP_API int64_t hmp_tensor_stride(const hmp_Tensor tensor, int64_t dim);
HMP_API int64_t hmp_tensor_nitems(const hmp_Tensor tensor);
HMP_API int64_t hmp_tensor_itemsize(const hmp_Tensor tensor);
HMP_API int64_t hmp_tensor_nbytes(const hmp_Tensor tensor);
HMP_API int hmp_tensor_dtype(const hmp_Tensor tensor);
HMP_API void *hmp_tensor_data(hmp_Tensor tensor);
HMP_API bool hmp_tensor_is_contiguous(hmp_Tensor tensor);
HMP_API int hmp_tensor_device_type(const hmp_Tensor tensor);
HMP_API int hmp_tensor_device_index(const hmp_Tensor tensor);

HMP_API hmp_Tensor hmp_tensor_clone(const hmp_Tensor tensor);
HMP_API hmp_Tensor hmp_tensor_alias(const hmp_Tensor tensor);
HMP_API hmp_Tensor hmp_tensor_view(const hmp_Tensor tensor,
                                   const int64_t *shape, int ndim);
HMP_API hmp_Tensor hmp_tensor_reshape(const hmp_Tensor tensor,
                                      const int64_t *shape, int ndim);

HMP_API hmp_Tensor hmp_tensor_slice(const hmp_Tensor tensor, int64_t dim,
                                    int64_t start, int64_t end, int64_t step);
HMP_API hmp_Tensor hmp_tensor_select(const hmp_Tensor tensor, int64_t dim,
                                     int64_t index);
HMP_API hmp_Tensor hmp_tensor_permute(const hmp_Tensor tensor,
                                      const int64_t *dims, int ndim);
HMP_API hmp_Tensor hmp_tensor_squeeze(const hmp_Tensor tensor, int64_t dim);
HMP_API hmp_Tensor hmp_tensor_unsqueeze(const hmp_Tensor tensor, int64_t dim);
HMP_API hmp_Tensor hmp_tensor_to_device(const hmp_Tensor data,
                                        const char *device, bool non_blocking);
HMP_API hmp_Tensor hmp_tensor_to_dtype(const hmp_Tensor data, int dtype);
HMP_API void hmp_tensor_copy_from(hmp_Tensor data, const hmp_Tensor from);

/////////////////// hmp_ColorModel ////////////
HMP_API hmp_ColorModel hmp_color_model(int cs, int cr, int cp, int ctc);
HMP_API void hmp_color_model_free(hmp_ColorModel cm);

HMP_API int hmp_color_model_space(const hmp_ColorModel cm);
HMP_API int hmp_color_model_range(const hmp_ColorModel cm);
HMP_API int hmp_color_model_primaries(const hmp_ColorModel cm);
HMP_API int hmp_color_model_ctc(const hmp_ColorModel cm);

/////////////////// hmp_PixelInfo ///////////////
HMP_API hmp_PixelInfo hmp_pixel_info(int format, const hmp_ColorModel cm);
HMP_API hmp_PixelInfo hmp_pixel_info_v1(int format, int cs, int cr);
HMP_API hmp_PixelInfo hmp_pixel_info_v2(int format, int cp, int ctc);
HMP_API void hmp_pixel_info_free(hmp_PixelInfo pix_info);

HMP_API int hmp_pixel_info_format(const hmp_PixelInfo pix_info);
HMP_API int hmp_pixel_info_space(const hmp_PixelInfo pix_info);
HMP_API int hmp_pixel_info_range(const hmp_PixelInfo pix_info);
HMP_API int hmp_pixel_info_primaries(const hmp_PixelInfo pix_info);
HMP_API int hmp_pixel_info_ctc(const hmp_PixelInfo pix_info);
HMP_API int hmp_pixel_info_infer_space(const hmp_PixelInfo pix_info);
HMP_API const hmp_ColorModel
hmp_pixel_info_color_model(const hmp_PixelInfo pix_info);
HMP_API bool hmp_pixel_info_is_rgbx(const hmp_PixelInfo pix_info);
HMP_API const char *hmp_pixel_info_stringfy(const hmp_PixelInfo pix_info,
                                            int *size);

/////////////////// hmp_Frame /////////////////
HMP_API hmp_Frame hmp_frame_make(int width, int height,
                                 const hmp_PixelInfo pix_info,
                                 const char *device);
HMP_API hmp_Frame hmp_frame_from_data(hmp_Tensor *data, int size,
                                      const hmp_PixelInfo pix_info);
HMP_API hmp_Frame hmp_frame_from_data_v1(hmp_Tensor *data, int size, int width,
                                         int height,
                                         const hmp_PixelInfo pix_info);
HMP_API void hmp_frame_free(hmp_Frame frame);
HMP_API bool hmp_frame_defined(const hmp_Frame frame);
HMP_API const hmp_PixelInfo hmp_frame_pix_info(const hmp_Frame frame);
HMP_API int hmp_frame_format(const hmp_Frame frame);
HMP_API int hmp_frame_width(const hmp_Frame frame);
HMP_API int hmp_frame_height(const hmp_Frame frame);
HMP_API int hmp_frame_dtype(const hmp_Frame frame);
HMP_API int hmp_frame_device_type(const hmp_Frame frame);
HMP_API int hmp_frame_device_index(const hmp_Frame frame);
HMP_API int64_t hmp_frame_nplanes(const hmp_Frame frame);
HMP_API const hmp_Tensor hmp_frame_plane(const hmp_Frame frame, int64_t p);
HMP_API hmp_Frame hmp_frame_to_device(const hmp_Frame frame, const char *device,
                                      bool non_blocking);
HMP_API void hmp_frame_copy_from(hmp_Frame self, const hmp_Frame from);
HMP_API hmp_Frame hmp_frame_clone(const hmp_Frame frame);
HMP_API hmp_Frame hmp_frame_crop(const hmp_Frame frame, int left, int top,
                                 int width, int height);
HMP_API hmp_Frame hmp_frame_reformat(const hmp_Frame frame,
                                     const hmp_PixelInfo pix_info);
HMP_API const char *hmp_frame_stringfy(const hmp_Frame frame, int *size);

#ifdef __cplusplus
} // extern "C"
#endif
