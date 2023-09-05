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

#include <bmf/sdk/common.h>
#include <bmf/sdk/hmp_import.h>
#include <bmf/sdk/sdk_interface.h>
#include <bmf/sdk/bmf_type_info.h>

namespace bmf_sdk {

class BMF_API VideoFrame : public OpaqueDataSet,
                           public SequenceData,
                           public Future {
    struct Private;

  public:
    using Frame = hmp::Frame;

    /**
     * @brief Construct a undefined Video Frame object
     *
     * @code {.cpp}
     * assert(VideoFrame() == false)
     * @endcode
     *
     */
    VideoFrame();

    /**
     * @brief VideoFrame is copyable
     *
     */
    VideoFrame(const VideoFrame &) = default;

    /**
     * @brief
     *
     * @return VideoFrame&
     */
    VideoFrame &operator=(const VideoFrame &) = default;

    /**
     * @brief VideoFrame is movable
     *
     */
    VideoFrame(VideoFrame &&) = default;

    /**
     * @brief Construct VideoFrame from Frame object
     *
     * @param frame Dedicated to YUV data, with irregular shapes between planes
     */
    VideoFrame(const Frame &frame);

    /**
     * @brief Construct VideoFrame(Frame) with given size (width, height),
     * PixelInfo,
     *  and device,
     *  for ease of use,  using factory function `VideoFrame::make(...)`
     *
     * @param width width of Y plane
     * @param height height of Y plane
     * @param pix_info PixelFormat and ColorModel
     * @param device device
     */
    VideoFrame(int width, int height, const PixelInfo &pix_info,
               const Device &device = kCPU);

    /**
     * @brief Facotry function to construct VideoFrame(Frame)
     *
     * @ref test_video_frame.cpp for more details
     *
     * @code {.cpp}
     *
     * //allocate VideoFrame with default device CPU
     * auto H420 = PixelInfo(PF_YUV420P, CS_BT607)
     * auto vf = VideoFrame::make(1920, 1080, H420);
     *
     * //allocate VideoFrame on CUDA device
     * auto vf = VideoFrame::make(1920, 1080, H420, kCUDA);
     * auto vf = VideoFrame::make(1920, 1080, H420, "cuda:0");
     * auto vf = VideoFrame::make(1920, 1080, H420, Device(kCUDA, 0));
     *
     * @endcode
     *
     * @tparam device
     *
     * @param width
     * @param height
     * @param format
     * @param device const char*, string, Device - infer to Device
     * @return VideoFrame
     */
    static VideoFrame make(int width, int height, const PixelInfo &pix_info,
                           const Device &device = kCPU) {
        return VideoFrame(width, height, pix_info, device);
    }

    static VideoFrame make(int width, int height, const PixelInfo &pix_info,
                           const std::string &device) {
        return VideoFrame(width, height, pix_info, device);
    }

    /**
     * @brief check if VideoFrame is defined
     *
     * @return true
     * @return false
     */
    operator bool() const;

    // image properties
    int width() const;
    int height() const;
    ScalarType dtype() const;

    /**
     * @brief
     *
     * @return const Frame&
     */
    const Frame &frame() const;

    /**
     * @brief Frame reformat, this only support rgb to yuv, or yuv to rgb
     *
     * @param pix_info
     * @return VideoFrame
     */
    VideoFrame reformat(const PixelInfo &pix_info);

    /**
     * @brief Return the selected region which specified by (x, y, w, h)
     *
     * @param x start col index
     * @param y start row index
     * @param w number of cols
     * @param h number of rows
     * @return VideoFrame
     */
    VideoFrame crop(int x, int y, int w, int h) const;

    // device related
    const Device &device() const override;
    VideoFrame cpu(bool non_blocking = false) const;
    VideoFrame cuda() const;

    /**
     * @brief In-place copy
     *
     * @param from data source which have the same type and shape
     * @return VideoFrame&
     */
    VideoFrame &copy_(const VideoFrame &from);

    /**
     * @brief Copy to target device,
     * if it have already reside on target device, shadow copy will be performed
     *
     * @param device Target device
     * @param non_blocking if true, internal allocator will try to allocate
     * pinned memory,
     * which can make data copy asynchronous
     * @return VideoFrame
     */
    VideoFrame to(const Device &device, bool non_blocking = false) const;

    /**
     * @brief copy all extra props(set by member func set_xxx) from
     * `from`(deepcopy if needed),
     *
     * @param from
     * @return VideoFrame&
     */
    VideoFrame &copy_props(const VideoFrame &from, bool copy_private = false);

    /**
     * @brief frame with multiple planes are stored in contiguous memory
     *
     * @param
     * @return VideoFrame
     */
    VideoFrame as_contiguous_storage();

  protected:
    VideoFrame(const std::shared_ptr<Private> &other);

  private:
    std::shared_ptr<Private> self;
};

} // namespace bmf_sdk

BMF_DEFINE_TYPE(bmf_sdk::VideoFrame)
