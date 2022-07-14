#include <hmp/core/logging.h>
#include <bmf/sdk/video_frame.h>

namespace bmf_sdk{


struct VideoFrame::Private
{
    Private(const Frame &frame_)
        : frame(frame_), is_image(false)
    {
    }

    Private(const Image &image_)
        : image(image_), is_image(true)
    {
    }

    Private(const Private &other) = default;

    //
    Image image;
    Frame frame;
    bool is_image = false;
};


VideoFrame::VideoFrame()
{
    //not defined
}


VideoFrame::VideoFrame(const Frame &frame)
{
    self = std::make_shared<Private>(frame);
}

VideoFrame::VideoFrame(const Image &image)
{
    self = std::make_shared<Private>(image);
}


VideoFrame::VideoFrame(int width, int height, const PixelInfo &pix_info, const Device &device)
    : VideoFrame(Frame(width, height, pix_info, device))
{
}

VideoFrame::VideoFrame(int width, int height, int channels, ChannelFormat format, const TensorOptions &options)
    : VideoFrame(Image(width, height, channels, format, options))
{
}

VideoFrame::VideoFrame(const std::shared_ptr<Private> &other)
    : self(other)
{
}

VideoFrame::operator bool() const
{
    return self.get() != nullptr;
}

int VideoFrame::width() const
{
    return self->is_image ? self->image.width() : self->frame.width();
}

int VideoFrame::height() const
{
    return self->is_image ? self->image.height() : self->frame.height();
}

ScalarType VideoFrame::dtype() const
{
    return self->is_image ? self->image.dtype() : self->frame.dtype();
}

bool VideoFrame::is_image() const
{
    return self->is_image;
}

const VideoFrame::Image& VideoFrame::image() const
{
    HMP_REQUIRE(self->is_image, "VideoFrame is not a image type");
    return self->image; //
}


const VideoFrame::Frame& VideoFrame::frame() const
{
    HMP_REQUIRE(!self->is_image, "VideoFrame is not a frame type");
    return self->frame;
}


VideoFrame VideoFrame::to_image(ChannelFormat format, bool contiguous) const
{
    Image image;
    if(self->is_image){
        image = self->image.to(format, contiguous);
    }
    else{
        image = self->frame.to_image(format);
    }

    VideoFrame vf(image);
    vf.copy_props(*this);
    return vf;
}


VideoFrame VideoFrame::to_frame(const PixelInfo &pix_info) const
{
    HMP_REQUIRE(self->is_image, "VideoFrame:to_frame require image type");

    auto frame = Frame::from_image(self->image, pix_info);
    VideoFrame vf(frame);
    vf.copy_props(*this);
    return vf;
}


VideoFrame VideoFrame::crop(int x, int y, int w, int h) const
{
    VideoFrame vf;
    if(self->is_image){
        auto image = self->image.crop(x, y, w, h);
        vf = VideoFrame(image);
    }
    else{
        auto frame = self->frame.crop(x, y, w, h);
        vf = VideoFrame(frame);
    }
    vf.copy_props(*this);
    return vf;
}


const Device& VideoFrame::device() const
{
    return self->is_image ? self->image.device() : self->frame.device();
}


VideoFrame VideoFrame::cpu(bool non_blocking) const
{
    VideoFrame vf;
    if(self->is_image){
        auto image = self->image.to(kCPU, non_blocking);
        vf = VideoFrame(image);
    }
    else{
        auto frame = self->frame.to(kCPU, non_blocking);
        vf = VideoFrame(frame);
    }
    vf.copy_props(*this);

    return vf;
}


VideoFrame VideoFrame::VideoFrame::cuda() const
{
    VideoFrame vf;
    if(self->is_image){
        auto image = self->image.to(kCUDA);
        vf = VideoFrame(image);
    }
    else{
        auto frame = self->frame.to(kCUDA);
        vf = VideoFrame(frame);
    }
    vf.copy_props(*this);

    return vf;
}


VideoFrame& VideoFrame::copy_(const VideoFrame &from)
{
    HMP_REQUIRE(from.is_image() == is_image(),
        "Can't copy frame to image or image to frame");

    if(self->is_image){
        self->image.copy_(from.image());
    }
    else{
        self->frame.copy_(from.frame());
    }

    return *this;
}

VideoFrame VideoFrame::to(const Device &device, bool non_blocking) const
{
    VideoFrame vf;
    if(self->is_image){
        auto image = self->image.to(device, non_blocking);
        vf = VideoFrame(image);
    }
    else{
        auto frame = self->frame.to(device, non_blocking);
        vf = VideoFrame(frame);
    }
    vf.copy_props(*this);

    return vf;
}

VideoFrame VideoFrame::to(ScalarType dtype) const
{
    VideoFrame vf;
    if(self->is_image){
        auto image = self->image.to(dtype);
        vf = VideoFrame(image);
    }
    else{
        HMP_REQUIRE(false, "VideoFrame: dtype convert is supported by Frame");
    }
    vf.copy_props(*this);

    return vf;
}


VideoFrame& VideoFrame::copy_props(const VideoFrame &from)
{
    OpaqueDataSet::copy_props(from);
    SequenceData::copy_props(from);
    Future::copy_props(from);
    return *this;
}

} //namespace bmf_sdk
