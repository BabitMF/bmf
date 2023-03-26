#pragma once

#include <bmf/sdk/common.h>
#include <bmf/sdk/hmp_import.h>
#include <bmf/sdk/sdk_interface.h>
#include <bmf/sdk/bmf_type_info.h>

namespace bmf_sdk{

class BMF_API VideoFrame : public OpaqueDataSet,
                   public SequenceData,
                   public Future
{
    struct Private;
public:
    using Image = hmp::Image; //RGB
    using Frame = hmp::Frame; //YUV
    
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
    VideoFrame(const VideoFrame&) = default;

    /**
     * @brief 
     * 
     * @return VideoFrame& 
     */
    VideoFrame& operator=(const VideoFrame&) = default;

    /**
     * @brief VideoFrame is movable
     * 
     */
    VideoFrame(VideoFrame&&) = default;

    /**
     * @brief Construct VideoFrame from Frame object
     * 
     * @param frame Dedicated to YUV data, with irregular shapes between planes
     */
    VideoFrame(const Frame &frame);

    /**
     * @brief Construct VideoFrame from Image object
     * 
     * @param image Dedicated to RGBx data, with unified shape
     */
    VideoFrame(const Image &image);

    /**
     * @brief Construct VideoFrame(Frame) with given size (width, height), PixelInfo, 
     *  and device,
     *  for ease of use,  using factory function `VideoFrame::make(...)`
     * 
     * @param width width of Y plane 
     * @param height height of Y plane
     * @param pix_info PixelFormat and ColorModel
     * @param device device
     */
    VideoFrame(int width, int height, const PixelInfo &pix_info, const Device &device = kCPU);

    /**
     * @brief Construct VideoFrame(Image) with give size (width, height), channels, ChannelFormat,
     * and Storage options(a.k.a TensorOptions), 
     * for ease of use, using factory functon `VideoFrame::make(...)`
     * 
     * @param width 
     * @param height 
     * @param channels 1 - Gray Image, 3 - RGB image, 4 - RGBA image
     * (User should take care the order of channels, e.g. RGB, or BGR) 
     * @param format (C, H, W) - kNCHW, (H, W, C) - kNHWC
     * @param options  Storage options include Device, ScalarType, and pinned_memory
     */
    VideoFrame(int width, int height, int channels = 3, ChannelFormat format = kNCHW, const TensorOptions &options = kUInt8);

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
    static VideoFrame make(int width, int height, const PixelInfo & pix_info, const Device &device = kCPU)
    {
        return VideoFrame(width, height, pix_info, device); 
    }

    static VideoFrame make(int width, int height, const PixelInfo & pix_info, const std::string &device)
    {
        return VideoFrame(width, height, pix_info, device); 
    }

    template<typename...Options>
    static VideoFrame make(int width, int height, int channels = 3, ChannelFormat format = kNCHW)
    {
        return VideoFrame(width, height, channels, format, 
            TensorOptions(kUInt8)); 
    }

    /**
     * @brief Factory function to construct VideoFrame(Image)
     * 
     * @ref test_video_frame.cpp for more details 
     * 
     * @code {.cpp}
     * 
     * //allocate VideoFrame with default options(CPU, uint8, 3 channels, kNCHW layout)
     * auto vf = VideoFrame::make(1920, 1080);
     * 
     * //allocate VideoFrame on CUDA device with dtype float
     * auto vf = VideoFrame::make(1920, 1080, 3, kNCHW, kCUDA, kFloat32);
     * auto vf = VideoFrame::make(1920, 1080, 3, kNCHW, "cuda:0", kFloat32);
     * auto vf = VideoFrame::make(1920, 1080, 3, kNCHW, Device(kCUDA, 0), kFloat32);
     * 
     * //allocate VideoFrame on CPU device with pinned_memory
     * auto vf = VideoFrame::make(1920, 1080, 3, kNCHW, true);
     * @endcode
     * 
     * @tparam Options 
     *  const char*, string, Device - infer to Device option
     *  ScalarType - infer to ScalarType option
     *  bool - infer to pinned_memory option
     * 
     * @param width 
     * @param height 
     * @param channels 
     * @param format 
     * @param opts 
     * @return VideoFrame 
     */
    template<typename...Options>
    static VideoFrame make(int width, int height, int channels, ChannelFormat format, Options&&...opts)
    {
        return VideoFrame(width, height, channels, format, 
            TensorOptions(kUInt8).options(std::forward<Options>(opts)...)); 
    }

    /**
     * @brief check if VideoFrame is defined 
     * 
     * @return true 
     * @return false 
     */
    operator bool() const;

    //image properties
    int width() const;
    int height() const;
    ScalarType dtype() const;

    /**
     * @brief check if internal data is Image or Frame
     * 
     * @return true 
     * @return false 
     */
    bool is_image() const;

    /**
     * @brief 
     * 
     * @return const Image& 
     */
    const Image &image() const;

    /**
     * @brief 
     * 
     * @return const Frame& 
     */
    const Frame &frame() const;

    /**
     * @brief Convert Frame to Image, or Image ChannelFormat convert
     * 
     * @param format 
     * @param contiguous ensure data storage is contiguous
     * @return VideoFrame 
     */
    VideoFrame to_image(ChannelFormat format = kNCHW, bool contiguous=true) const;

    /**
     * @brief Convert Image to Frame
     * 
     * @param format 
     * @return VideoFrame 
     */
    VideoFrame to_frame(const PixelInfo &pix_info) const;

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
    const Device& device() const override;
    VideoFrame cpu(bool non_blocking = false) const;
    VideoFrame cuda() const;

    /**
     * @brief In-place copy
     * 
     * @param from data source which have the same type and shape
     * @return VideoFrame& 
     */
    VideoFrame& copy_(const VideoFrame &from);

    /**
     * @brief Copy to target device, 
     * if it have already reside on target device, shadow copy will be performed
     * 
     * @param device Target device
     * @param non_blocking if true, internal allocator will try to allocate pinned memory, 
     * which can make data copy asynchronous
     * @return VideoFrame 
     */
    VideoFrame to(const Device &device, bool non_blocking = false) const;

    /**
     * @brief Convert to target dtype
     * 
     * @param dtype 
     * @return VideoFrame 
     */
    VideoFrame to(ScalarType dtype) const;

    /**
     * @brief copy all extra props(set by member func set_xxx) from `from`(deepcopy if needed), 
     * 
     * @param from 
     * @return VideoFrame& 
     */
    VideoFrame& copy_props(const VideoFrame &from);

protected:
    VideoFrame(const std::shared_ptr<Private> &other);

private:
    std::shared_ptr<Private> self;
};

} //namespace bmf_sdk


BMF_DEFINE_TYPE(bmf_sdk::VideoFrame)