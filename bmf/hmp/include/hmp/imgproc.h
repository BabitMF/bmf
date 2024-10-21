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

#include "imgproc/formats.h"
#include <hmp/imgproc/image.h>

namespace hmp {
namespace img {

HMP_API TensorList frame_format(const TensorList &data, PixelFormat format,
                                int width, int height, bool has_batch = false);
HMP_API TensorList frame_format(const TensorList &data,
                                const PixelFormatDesc &pix_desc, int width,
                                int height, bool has_batch = false);
HMP_API TensorList frame_format(const TensorList &data, PixelFormat format,
                                bool has_batch = false);

HMP_API Tensor image_format(const Tensor &image, ChannelFormat cformat = kNHWC,
                            bool batch_first = true);
HMP_API TensorList image_format(const TensorList &images,
                                ChannelFormat cformat = kNHWC,
                                bool batch_first = true);

HMP_API Tensor &yuv_to_rgb(Tensor &dst, const TensorList &src,
                           const PixelInfo &pix_info,
                           ChannelFormat cformat = kNCHW);
HMP_API Tensor yuv_to_rgb(const TensorList &src, const PixelInfo &pix_info,
                          ChannelFormat cformat = kNCHW);
HMP_API Tensor &yuv_to_bgr(Tensor &dst, const TensorList &src,
                           const PixelInfo &pix_info,
                           ChannelFormat cformat = kNCHW);
HMP_API Tensor yuv_to_bgr(const TensorList &src,
                          const PixelInfo &pix_info,
                          ChannelFormat cformat = kNCHW);

HMP_API TensorList &rgb_to_yuv(TensorList &dst, const Tensor &src,
                               const PixelInfo &pix_info,
                               ChannelFormat cformat = kNCHW);
HMP_API TensorList rgb_to_yuv(const Tensor &src, const PixelInfo &pix_info,
                              ChannelFormat cformat = kNCHW);
HMP_API TensorList &bgr_to_yuv(TensorList &dst, const Tensor &src,
                               const PixelInfo &pix_info,
                               ChannelFormat cformat = kNCHW);
HMP_API TensorList bgr_to_yuv(const Tensor &src,
                              const PixelInfo &pix_info,
                              ChannelFormat cformat = kNCHW);

HMP_API TensorList &yuv_to_yuv(TensorList &dst, const TensorList &src,
                               const PixelInfo &dst_pix_info,
                               const PixelInfo &src_pix_info);
HMP_API TensorList yuv_to_yuv(const TensorList &src,
                              const PixelInfo &dst_pix_info,
                              const PixelInfo &src_pix_info);

HMP_API TensorList &
yuv_resize(TensorList &dst, const TensorList &src, const PixelInfo &pix_info,
           ImageFilterMode mode = ImageFilterMode::Bilinear);
HMP_API TensorList &yuv_rotate(TensorList &dst, const TensorList &src,
                               const PixelInfo &pix_info,
                               ImageRotationMode rotate);
HMP_API TensorList &yuv_mirror(TensorList &dst, const TensorList &src,
                               const PixelInfo &pix_info, ImageAxis axis);

HMP_API Tensor &resize(Tensor &dst, const Tensor &src,
                       ImageFilterMode mode = ImageFilterMode::Bilinear,
                       ChannelFormat cformat = kNCHW);
HMP_API Tensor resize(const Tensor &src, int width, int height,
                      ImageFilterMode mode = ImageFilterMode::Bilinear,
                      ChannelFormat cformat = kNCHW);
HMP_API Tensor &rotate(Tensor &dst, const Tensor &src,
                       ImageRotationMode mode = ImageRotationMode::Rotate90,
                       ChannelFormat cformat = kNCHW);
HMP_API Tensor rotate(const Tensor &src,
                      ImageRotationMode mode = ImageRotationMode::Rotate90,
                      ChannelFormat cformat = kNCHW);
HMP_API Tensor &mirror(Tensor &dst, const Tensor &src,
                       ImageAxis axis = ImageAxis::Vertical,
                       ChannelFormat cformat = kNCHW);
HMP_API Tensor mirror(const Tensor &src, ImageAxis axis = ImageAxis::Vertical,
                      ChannelFormat cformat = kNCHW);

HMP_API Tensor normalize(const Tensor &src, const Tensor &mean,
                         const Tensor &std, ChannelFormat cformat = kNCHW);
HMP_API Tensor &normalize(Tensor &dst, const Tensor &src, const Tensor &mean,
                          const Tensor &std, ChannelFormat cformat = kNCHW);

//
HMP_API Tensor &erode(Tensor &dst, const Tensor &src,
                      const optional<Tensor> &kernel = nullopt,
                      ChannelFormat cformat = kNCHW);
HMP_API Tensor erode(const Tensor &src,
                     const optional<Tensor> &kernel = nullopt,
                     ChannelFormat cformat = kNCHW);
HMP_API Tensor &dilate(Tensor &dst, const Tensor &src,
                       const optional<Tensor> &kernel = nullopt,
                       ChannelFormat cformat = kNCHW);
HMP_API Tensor dilate(const Tensor &src,
                      const optional<Tensor> &kernel = nullopt,
                      ChannelFormat cformat = kNCHW);

// Edge detection
HMP_API Tensor &sobel(Tensor &dst, const Tensor &src, int64_t dx, int64_t dy,
                      int64_t ksize = 3, const Scalar &scale = 1,
                      const Scalar &delta = 0, ChannelFormat cformat = kNCHW);
HMP_API Tensor sobel(const Tensor &src, int64_t dx, int64_t dy,
                     int64_t ksize = 3, const Scalar &scale = 1,
                     const Scalar &delta = 0, ChannelFormat cformat = kNCHW);

HMP_API Tensor &canny(Tensor &dst, const Tensor &src, const Scalar &low_thresh,
                      const Scalar &high_thresh, int64_t aperture_size = 3,
                      bool l2_gradient = false, ChannelFormat cformat = kNCHW);
HMP_API Tensor canny(const Tensor &src, const Scalar &low_thresh,
                     const Scalar &high_thresh, int64_t aperture_size = 3,
                     bool l2_gradient = false, ChannelFormat cformat = kNCHW);

// Filters
HMP_API Tensor &filter2d(Tensor &dst, const Tensor &src, const Tensor &kernel,
                         const Scalar &delta = 0,
                         ChannelFormat cformat = kNCHW);
HMP_API Tensor filter2d(const Tensor &src, const Tensor &kernel,
                        const Scalar &delta = 0, ChannelFormat cformat = kNCHW);

HMP_API Tensor warp_perspective(const Tensor &src, int64_t width,
                                int64_t height, const Tensor &M,
                                ImageFilterMode mode = kBicubic,
                                ChannelFormat cformat = kNCHW);
HMP_API Tensor &warp_perspective(Tensor &dst, const Tensor &src,
                                 const Tensor &M,
                                 ImageFilterMode mode = kBicubic,
                                 ChannelFormat cformat = kNCHW);

HMP_API Tensor bilateral_filter(const Tensor &src, int d,
                                const Scalar &sigma_color,
                                const Scalar &sigma_space,
                                ChannelFormat cformat = kNCHW);
HMP_API Tensor &bilateral_filter(Tensor &dst, const Tensor &src, int d,
                                 const Scalar &sigma_color,
                                 const Scalar &sigma_space,
                                 ChannelFormat cformat = kNCHW);

HMP_API Tensor gaussian_blur(const Tensor &src, int kx, int ky,
                             const Scalar &sigma_x, const Scalar &sigma_y = 0,
                             ChannelFormat cformat = kNCHW);
HMP_API Tensor &gaussian_blur(Tensor &dst, const Tensor &src, int kx, int ky,
                              const Scalar &sigma_x, const Scalar &sigma_y = 0,
                              ChannelFormat cformat = kNCHW);

// dst = src0 * (1 - alpha) + src1 * (alpha);
HMP_API Tensor &overlay(Tensor &dst, const Tensor &src0, const Tensor &src1,
                        const Tensor &alpha);
HMP_API Tensor overlay(const Tensor &src0, const Tensor &src1,
                       const Tensor &alpha);

HMP_API Tensor transfer(const Tensor &src, const ChannelFormat &src_format,
                        const ChannelFormat &dst_format);
} // namespace img
} // namespace hmp
