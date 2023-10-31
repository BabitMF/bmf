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

#include <hmp/tensor.h>
#include <hmp/imgproc/formats.h>

namespace hmp {

// class Image;

class HMP_API Frame {
  public:
    Frame() = default;
    Frame(const Frame &) = default;

    Frame(int width, int height, const PixelInfo &pix_info,
          const Device &device = kCPU);

    Frame(const TensorList &data, int width, int height,
          const PixelInfo &pix_info, const Tensor &storage_tensor = {});
    Frame(const TensorList &data, const PixelInfo &pix_info,
          const Tensor &storage_tensor = {});
    Frame(const Tensor &data, const PixelInfo &pix_info,
          const Tensor &storage_tensor = {});

    /**
     * @brief check is this frame is defined
     *
     * @return true
     * @return false
     */
    operator bool() const { return data_.size() > 0; }

    /**
     * @brief return PixelFormat description information
     *
     * @return const PixelFormatInfo&
     */
    const PixelFormatDesc &pix_desc() const { return pix_desc_; }

    const PixelInfo &pix_info() const { return pix_info_; }

    /**
     * @brief
     *
     * @return PixelFormat
     */
    PixelFormat format() const { return pix_info_.format(); }

    /**
     * @brief width of frist plane
     *
     * @return int
     */
    int width() const { return width_; }

    /**
     * @brief height of first plane
     *
     * @return int
     */
    int height() const { return height_; }

    /**
     * @brief data type of the internal data
     *
     * @return ScalarType
     */
    ScalarType dtype() const { return plane(0).dtype(); }

    /**
     * @brief
     *
     * @return Device
     */
    const Device &device() const { return plane(0).device(); }

    /**
     * @brief number of planes
     *
     * @return int64_t
     */
    int64_t nplanes() const { return data_.size(); }

    /**
     * @brief
     *
     * @param p
     * @return const Tensor&
     */
    const Tensor &plane(int64_t p) const { return data_[p]; }
    Tensor &plane(int64_t p) { return data_[p]; }

    /**
     * @brief return raw data pointer
     *
     * @param p
     * @return void*
     */
    void *plane_data(int64_t p) { return plane(p).unsafe_data(); }
    const void *plane_data(int64_t p) const { return plane(p).unsafe_data(); }

    /**
     * @brief
     *
     * @return const TensorList&
     */
    const TensorList &data() const { return data_; }
    TensorList &data() { return data_; }

    /**
     * @brief copy data to target device, if the data have already reside in
     * target device,
     *  shadow copy will be performed
     *
     * @param device target device
     * @param non_blocking if true, it will try copy data asynchronously
     *                     otherwise, the lib will ensure the data is ready
     * before return
     * @return Frame
     *
     */
    Frame to(const Device &device, bool non_blocking = false) const;

    /**
     * @brief
     *
     * @param device
     * @param non_blocking
     * @return Frame
     */
    Frame to(DeviceType device, bool non_blocking = false) const;

    /**
     * @brief inplace copy
     *
     * @param from
     * @return Frame&
     */
    Frame &copy_(const Frame &from);

    /**
     * @brief clone current frame(deep copy)
     *
     * @return Frame
     */
    Frame clone() const;

    /**
     * @brief select the region(ROI)
     *
     * @param left
     * @param top
     * @param width
     * @param height
     * @return Frame
     */
    Frame crop(int left, int top, int width, int height) const;

    /**
     * @brief convert Frame format
     *
     * @param &pix_info
     * @return Frame
     */
    Frame reformat(const PixelInfo &pix_info);
    const Tensor &storage() const { return storage_tensor_; }

    /**
     * @brief frame with multiple planes are stored in contiguous memory
     *
     * @param
     * @return Frame
     */
    Frame as_contiguous_storage();

  private:
    int width_, height_;
    PixelFormatDesc pix_desc_;
    PixelInfo pix_info_;
    TensorList data_;

    /// Each plane of the frame is represented using a tensor
    /// and we want the memory of multi-planes frame to be allocated
    /// continuously. This requires an additional member variable to represent a
    /// contiguous storage space.
    Tensor storage_tensor_;
};

HMP_API std::string stringfy(const Frame &frame);
TensorList from_storage_tensor(const Tensor &storage_tensor,
                               const TensorList &mirror);

} // namespace hmp
