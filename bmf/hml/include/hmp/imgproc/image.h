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

namespace hmp{

class Image;

class HMP_API Frame
{
public:
    Frame() = default;
    Frame(const Frame&) = default;

    Frame(int width, int height, const PixelInfo &pix_info, const Device &device = kCPU);

    Frame(const TensorList &data, int width, int height, const PixelInfo &pix_info);
    Frame(const TensorList &data, const PixelInfo &pix_info);

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
    const PixelFormatDesc& pix_desc() const { return pix_desc_; }

    const PixelInfo& pix_info() const { return pix_info_; }

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
    const Tensor &plane(int64_t p) const  { return data_[p]; }
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
     * @brief copy data to target device, if the data have already reside in target device,
     *  shadow copy will be performed
     * 
     * @param device target device
     * @param non_blocking if true, it will try copy data asynchronously
     *                     otherwise, the lib will ensure the data is ready before return
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
     * @brief convert to Image
     * 
     * @param cformat
     * @return Image 
     */
    Image to_image(ChannelFormat cformat = kNCHW) const;

    /**
     * @brief convert from Image
     * 
     * @param image 
     * @param pformat 
     * @return Frame 
     */
    static Frame from_image(const Image &image, const PixelInfo &pix_info);


private:
    int width_, height_;
    PixelFormatDesc pix_desc_;
    PixelInfo pix_info_;
    TensorList data_;
};


class HMP_API Image
{
public:
    Image() = default;
    Image(const Image&) = default;

    /**
     * @brief Construct a new Image object
     * 
     * @param data 
     * @param format 
     * @param cm 
     */
    Image(const Tensor &data, ChannelFormat format, const ColorModel &cm);

    /**
     * @brief Construct a new Image object
     * 
     * @param data image data 2(HxW) or 3(HxWxC) dims
     * @param format layout of the image data
     * @param cm pixel color model
     */
    Image(const Tensor &data, ChannelFormat format);

    /**
     * @brief Construct a new Image object
     * 
     * @param width 
     * @param height 
     * @param channels 1, 2, 3, 4
     * @param format 
     * @param cm
     * @param options
     */
    Image(int width, int height, int channels, ChannelFormat format = kNCHW, const TensorOptions &options = kUInt8);

    /**
     * @brief return if it is a valid Image
     * 
     * @return true 
     * @return false 
     */
    operator bool() const { return data_.defined(); }

    /**
     * @brief 
     * 
     * @return ChannelFormat 
     */
    ChannelFormat format() const { return format_; }

    /**
     * @brief Set the color model object
     * 
     * @param cm 
     */
    void set_color_model(const ColorModel &cm) { cm_ = cm; }

    /**
     * @brief 
     * 
     * @return const ColorModel& 
     */
    const ColorModel &color_model() const { return cm_; }

    /**
     * @brief width dim index
     * 
     * @return int 
     */
    int wdim() const { return hdim() + 1; }

    /**
     * @brief height dim index
     * 
     * @return int 
     */
    int hdim() const { return format_ == kNCHW ? 1 : 0; }

    /**
     * @brief channel dim index
     * 
     * @return int 
     */
    int cdim() const { return format_ == kNCHW ? 0 : 2; }

    /**
     * @brief width of image
     * 
     * @return int 
     */
    int width() const { return data_.size(wdim()); }

    /**
     * @brief height of image
     * 
     * @return int 
     */
    int height() const { return data_.size(hdim()); } 

    /**
     * @brief number of channels
     * 
     * @return int 
     */
    int nchannels() const { return data_.size(cdim()); }

    /**
     * @brief convert to target dtype, if image's dtype is same as target dtype,
     * shadow copy will be performed
     * 
     * @return ScalarType 
     */
    ScalarType dtype() const { return data_.dtype(); }

    /**
     * @brief 
     * 
     * @return const Device& 
     */
    const Device &device() const { return data_.device(); }

    /**
     * @brief 
     * 
     * @return const Tensor& 
     */
    const Tensor &data() const { return data_; }
    Tensor &data() { return data_; }

    /**
     * @brief return raw data pointer 
     * 
     * @return const void* 
     */
    const void* unsafe_data() const { return data().unsafe_data(); }
    void* unsafe_data() { return data().unsafe_data(); }

    /**
     * @brief copy data to target device, if data has already reside in target 
     * device, shadow copy will be performed
     * 
     * @param device target device
     * @param non_blocking 
     * @return Image 
     */
    Image to(const Device &device, bool non_blocking = false) const;

    /**
     * @brief 
     * 
     * @param device 
     * @param non_blocking 
     * @return Image 
     */
    Image to(DeviceType device, bool non_blocking = false) const;

    /**
     * @brief 
     * 
     * @param dtype 
     * @return Image 
     */
    Image to(ScalarType dtype) const;

    /**
     * @brief convert to target channel format, if format has already satisfied,
     * shadow copy will be performed
     * 
     * @param format target channel format
     * @param contiguous ensure data is contiguous or not, like other functions,
     * if data is already contiguous, shadow copy will performed
     * @return Image 
     */
    Image to(ChannelFormat format, bool contiguous = true) const;

    /**
     * @brief copy from another image
     * 
     * @param from 
     * @return Image& 
     */
    Image &copy_(const Image &from);

    /**
     * @brief 
     * 
     * @return Image 
     */
    Image clone() const;

    /**
     * @brief 
     * 
     * @param left 
     * @param top 
     * @param width 
     * @param height 
     * @return Image 
     */
    Image crop(int left, int top, int width, int height) const;

    /**
     * @brief
     * 
     * @param channel 
     * @return Image 
     */
    Image select(int channel) const;


private:
    ColorModel cm_;
    ChannelFormat format_;
    Tensor data_;
};


HMP_API std::string stringfy(const Frame &frame);
HMP_API std::string stringfy(const Image &image);

} //namespace hmp

