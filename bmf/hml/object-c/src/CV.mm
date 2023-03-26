
#import <hmp/oc/CV.h>
#import <hmp/oc/Metal.h>
#include <vector>
#import <OpenGLES/EAGL.h>
#import <OpenGLES/ES2/gl.h>

#if !__has_feature(objc_arc)
#error "Source must compile with ARC enabled!!"
#endif


namespace hmp{
namespace oc{

#define HMP_FORALL_CVPIXEL_FORMATS(_)  \
    _(PF_YUV420P, kCVPixelFormatType_420YpCbCr8Planar)              \
    _(PF_NV12, kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange)     \
    _(PF_RGB24, kCVPixelFormatType_24RGB)                           \
    _(PF_BGRA32, kCVPixelFormatType_32BGRA)


int toCVPixelFormat(PixelFormat format, ColorRange range)
{
    if(format == PF_NV12 && range == CR_JPEG){
        return kCVPixelFormatType_420YpCbCr8BiPlanarFullRange;
    }

    switch(format){
#define CASE(f, cvf) case f: return cvf;
        HMP_FORALL_CVPIXEL_FORMATS(CASE)
#undef CASE
        default:
            throw std::runtime_error("unsupported PixelFromat " + std::to_string(format));
    }
}

PixelFormat fromCVPixelFormat(int cvFormat)
{
    if(cvFormat == kCVPixelFormatType_420YpCbCr8BiPlanarFullRange){
        return PF_NV12;
    }

    switch(cvFormat){
#define CASE(f, cvf) case cvf: return f;
        HMP_FORALL_CVPIXEL_FORMATS(CASE)
#undef CASE
        default:
            throw std::runtime_error("unsupported CVPixelFormat " + std::to_string(cvFormat));
    }

}


struct PixelBuffer::Private
{
    CVPixelBufferRef pb = nil;

    CVOpenGLESTextureCacheRef glESCache = nil;
    std::vector<CVOpenGLESTextureRef> glTexs;

    CVMetalTextureCacheRef mtlCache = nil;
    std::vector<CVMetalTextureRef> mtlTexs;

    ~Private()
    {
        for(auto &t : glTexs){
            CVBufferRelease(t);
        }
        glTexs.clear();
        
        for(auto &t : mtlTexs){
            CVBufferRelease(t);
        }
        mtlTexs.clear();
        
        if(glESCache){
            CVOpenGLESTextureCacheFlush(glESCache, 0);
            CFRelease(glESCache);
        }

        if(mtlCache){
            CVMetalTextureCacheFlush(mtlCache, 0);
            CFRelease(mtlCache);
        }

        if(pb){
            CVPixelBufferRelease(pb);
        }
    }

};


PixelBuffer::PixelBuffer(CVPixelBufferRef pixel_buffer)
{   
    self = std::make_shared<Private>();
    self->pb = pixel_buffer;
}

CVPixelBufferRef PixelBuffer::buffer()
{
    return self->pb;
}

PixelBuffer::PixelBuffer(void *pixel_buffer)
{
    self = std::make_shared<Private>();
    self->pb = (CVPixelBufferRef)pixel_buffer;
}

const void* PixelBuffer::handle() const
{
    return self.get() != nullptr ? self->pb : nullptr;
}


PixelBuffer::PixelBuffer(int width, int height, PixelFormat format, ColorRange range, bool gl, bool metal)
{
    auto cvPixelFormat = toCVPixelFormat(format, range);

    NSDictionary* cvBufferProperties = @{
        (__bridge NSString*)kCVPixelBufferOpenGLCompatibilityKey : gl ? @YES : @NO,
        (__bridge NSString*)kCVPixelBufferMetalCompatibilityKey : metal ? @YES : @NO,
    };

    CVPixelBufferRef pixelBuffer;
    CVReturn cvret = CVPixelBufferCreate(kCFAllocatorDefault,
                        width, height,
                        cvPixelFormat,
                        (__bridge CFDictionaryRef)cvBufferProperties,
                        &pixelBuffer);
    if(cvret != 0){
        throw std::runtime_error("create CVPixelBuffer failed with format "
                                 + std::to_string(format) + ", " + std::to_string(cvret));
    }

    self = std::make_shared<Private>();
    self->pb = pixelBuffer;
}

unsigned PixelBuffer::createGlTexture(int plane, void* context) const
{
    if(self->glESCache == nil){
        auto cvret = CVOpenGLESTextureCacheCreate(kCFAllocatorDefault,
                    nil,
                    (__bridge CVEAGLContext)context,
                    nil,
                    &self->glESCache);
        if(cvret != 0){
            throw std::runtime_error("create OpenGLESTextureCache failed, " + std::to_string(cvret));
        }
    }

    int width = CVPixelBufferGetWidth(self->pb);
    int height = CVPixelBufferGetHeight(self->pb);
    auto cvPixelFormat = CVPixelBufferGetPixelFormatType(self->pb);
    PixelFormatDesc desc(fromCVPixelFormat(cvPixelFormat));

    unsigned glFormat;
    unsigned glInternalFormat;
    unsigned glType = GL_UNSIGNED_BYTE;
    if(desc.format() == PF_NV12){
        if(plane == 0){
            glFormat = GL_LUMINANCE;
            glInternalFormat = GL_LUMINANCE;
        }
        else{
            glFormat = GL_LUMINANCE_ALPHA;
            glInternalFormat = GL_LUMINANCE_ALPHA;
        }
    }
    else if(desc.format() == PF_BGRA32){
        glFormat = GL_RGBA;
        glInternalFormat = GL_RGBA;
    }
    else{
        throw std::runtime_error("unsupported PixelFormat when convert to gl texture "
                                  + std::to_string(desc.format()));
    }
    
    CVOpenGLESTextureRef glTex = nil;
    CVReturn cvret = CVOpenGLESTextureCacheCreateTextureFromImage(kCFAllocatorDefault,
                self->glESCache,
                self->pb,
                nil,
                GL_TEXTURE_2D,
                glInternalFormat,
                desc.infer_width(width, plane),
                desc.infer_height(height, plane),
                glFormat,
                glType,
                plane,
                &glTex);
    if(cvret != 0){
        throw std::runtime_error("create opengl texture from CVPixelBuffer failed, " + std::to_string(cvret));
    }

    self->glTexs.push_back(glTex);

    return CVOpenGLESTextureGetName(glTex);
}

metal::Texture PixelBuffer::createMetalTexture(int plane)
{
    if(self->mtlCache == nil){
        auto device = metal::Device::current().device();
        if(device == nil){
            throw std::runtime_error("No metal device is specified");
        }
        auto cvret = CVMetalTextureCacheCreate(
                        kCFAllocatorDefault,
                        nil,
                        device,
                        nil,
                        &self->mtlCache);
        if(cvret != 0){
            throw std::runtime_error("Create metal texture cache failed, " + std::to_string(cvret));
        }
    }
    
    int width = CVPixelBufferGetWidth(self->pb);
    int height = CVPixelBufferGetHeight(self->pb);
    auto cvPixelFormat = CVPixelBufferGetPixelFormatType(self->pb);
    PixelFormatDesc desc(fromCVPixelFormat(cvPixelFormat));

    MTLPixelFormat mtlFormat;
    if(desc.format() == PF_GRAY8 || desc.format() == PF_YUV420P){
        mtlFormat = MTLPixelFormatR8Unorm;
    }
    else if(desc.format() == PF_NV12){
        if(plane == 0){
            mtlFormat = MTLPixelFormatR8Unorm;
        }
        else{
            mtlFormat = MTLPixelFormatRG8Unorm;
        }
    }
    else if(desc.format() == PF_BGRA32){
        mtlFormat = MTLPixelFormatBGRA8Unorm;
    }
    else{
        throw std::runtime_error("unsupported PixelFormat when convert to gl texture "
                                  + std::to_string(desc.format()));
    }

    //
    CVMetalTextureRef mtlTex;
    auto cvret = CVMetalTextureCacheCreateTextureFromImage(
                kCFAllocatorDefault,
                self->mtlCache,
                self->pb, nil,
                mtlFormat,
                desc.infer_width(width, plane),
                desc.infer_height(height, plane),
                plane,
                &mtlTex);
    if(cvret != 0){
        throw std::runtime_error("Create metal texture from CVPixelBuffer failed, " + std::to_string(cvret));
    }

    self->mtlTexs.push_back(mtlTex);

    return metal::Texture(CVMetalTextureGetTexture(mtlTex));

}


int PixelBuffer::width() const
{
    return CVPixelBufferGetWidth(self->pb);

}
int PixelBuffer::height() const
{
    return CVPixelBufferGetHeight(self->pb);
}

int PixelBuffer::format() const
{
    auto cvPixelFormat = CVPixelBufferGetPixelFormatType(self->pb);
    return fromCVPixelFormat(cvPixelFormat);
}

ColorRange PixelBuffer::range() const
{
    auto cvPixelFormat = CVPixelBufferGetPixelFormatType(self->pb);
    if(cvPixelFormat == kCVPixelFormatType_420YpCbCr8BiPlanarFullRange){
        return CR_JPEG;
    }
    else{
        return CR_MPEG;
    }
}



}} //
