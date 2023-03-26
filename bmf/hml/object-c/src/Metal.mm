
#import <Metal/Metal.h>
#import <hmp/oc/Metal.h>
#import <Metal/MTLTexture.h>

namespace hmp{
namespace metal{


#define HMP_FORALL_MTLPIXEL_FORMATS(_)               \
    _(PF_GRAY8, MTLPixelFormatR8Unorm)              \
    _(PF_YA8,  MTLPixelFormatRG8Unorm)              \
    _(PF_RGBA32, MTLPixelFormatRGBA8Unorm)          \
    _(PF_BGRA32, MTLPixelFormatBGRA8Unorm)


unsigned toMTLPixelFormat(PixelFormat format)
{
    switch(format){
#define CASE(f, cvf) case f: return cvf;
        HMP_FORALL_MTLPIXEL_FORMATS(CASE)
#undef CASE
        default:
            throw std::runtime_error("unsupported PixelFromat " + std::to_string(format));
    }
}

PixelFormat fromMTLPixelFormat(unsigned mtlFormat)
{
    switch(mtlFormat){
#define CASE(f, cvf) case cvf: return f;
        HMP_FORALL_MTLPIXEL_FORMATS(CASE)
#undef CASE
        default:
            throw std::runtime_error("unsupported MTLPixelFormat " + std::to_string(mtlFormat));
    }
}


//
struct Texture::Private{
    id<MTLTexture> tex;
};

Texture::Texture(id<MTLTexture> texture)
{
    self = std::make_shared<Private>();
    self->tex = texture;
}

Texture::Texture(void *texture)
{
    self = std::make_shared<Private>();
    self->tex = (__bridge id<MTLTexture>)texture;
}

Texture::Texture(int width, int height, PixelFormat format, unsigned usage, bool mipmapped)
{
     MTLPixelFormat mtlFormat = (MTLPixelFormat)toMTLPixelFormat(format);
     MTLTextureDescriptor *texDesc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:mtlFormat
                                                                                              width:width
                                                                                                height:height
                                                                                             mipmapped:mipmapped];
    texDesc.usage = 0;
    if(usage & kRead){
        texDesc.usage |= MTLTextureUsageShaderRead;
    }
    if(usage & kWrite){
        texDesc.usage |= MTLTextureUsageShaderWrite;
    }

    id<MTLDevice> device = Device::current().device();
    if(!device){
        throw std::runtime_error("No device is specified to create MTLTexture");
    }
    
    id<MTLTexture> tex = [device newTextureWithDescriptor:texDesc];
    if(tex){
        self = std::make_shared<Private>();
        self->tex = tex;
    }
    else{
        throw std::runtime_error("create MTLTexture failed");
    }
}


id<MTLTexture> Texture::texture()
{
    return self.get() != nil ? self->tex : nil;
}


const void* Texture::handle() const
{
    return self.get() != nil ? (__bridge const void*)self->tex : nil;
}

int Texture::width() const
{
    return (int)[self->tex width];
}

int Texture::height() const
{
    return (int)[self->tex height];
}

int Texture::depth() const
{
    return (int)[self->tex depth];
}


PixelFormat Texture::format() const
{
    return fromMTLPixelFormat(pixel_format());
}

int Texture::texture_type() const
{
    return (int)[self->tex textureType];
}

int Texture::pixel_format() const
{
    return (int)[self->tex pixelFormat];
}

int Texture::sample_count() const
{
    return (int)[self->tex sampleCount];
}


bool Texture::read(void *data, int bytesPerRow)
{
    MTLRegion region = MTLRegionMake2D(0, 0, width(), height());
    [self->tex getBytes:data bytesPerRow:bytesPerRow fromRegion:region mipmapLevel:0];
    return true;
}


bool Texture::write(const void *data, int bytesPerRow)
{
    MTLRegion region = MTLRegionMake2D(0, 0, width(), height());
    [self->tex replaceRegion:region mipmapLevel:0 withBytes:(void*)data bytesPerRow:bytesPerRow];
    return true;
}

int Texture::max_width()
{
    //https://stackoverflow.com/questions/58366416/how-to-get-programmatically-the-maximum-texture-size-width-and-height
    int maxw = 4096;
    id<MTLDevice> mtldevice = Device::current().device();
    if(mtldevice == nil){
        throw std::runtime_error("No MTLDevice is selected");
    }

    if ([mtldevice supportsFeatureSet:MTLFeatureSet_iOS_GPUFamily4_v1] || [mtldevice supportsFeatureSet:MTLFeatureSet_iOS_GPUFamily3_v1]) {
        maxw = 16384;
    }else if ([mtldevice supportsFeatureSet:MTLFeatureSet_iOS_GPUFamily2_v2] || [mtldevice supportsFeatureSet:MTLFeatureSet_iOS_GPUFamily1_v2]) {
        maxw = 8192;
    } else {
        maxw = 4096;
    }
    return maxw;
}

int Texture::max_height()
{
    return max_width();
}

//

static thread_local Device sCurrentDevice;

struct Device::Private{
    id<MTLDevice> dev;
};


Device::Device(id<MTLDevice> device)
{
    self = std::make_shared<Private>();
    self->dev = device;
}

id<MTLDevice> Device::device()
{
    return self.get() != nil ? self->dev : nil;
}

Device::Device(void *device)
{
    self->dev = (__bridge id<MTLDevice>)device;
}

const void* Device::handle() const
{
    return (__bridge const void*)self->dev;
}


Device& Device::current()
{
    return sCurrentDevice;
}

void Device::set_current(const Device &dev)
{
    sCurrentDevice = dev;
}


}} //namespace
