/*
 * Copyright 2024 Babit Authors
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

#ifndef _BMFLITE_DEMO_METAL_HELPER_H_
#define _BMFLITE_DEMO_METAL_HELPER_H_

#import "BmfLiteDemoMacro.h"
#import <AVFoundation/AVFoundation.h>
#import <Metal/Metal.h>
#include <atomic>

BMFLITE_DEMO_NAMESPACE_BEGIN

struct MetalFormat {
    MetalFormat() {}
    enum Format {
        RGB_8 = 0,
        RGB_10 = 1,
        NV12_8 = 2,
        NV12_10 = 3,
        AUTO,
    };
    MetalFormat(int fmt, bool video_range = true) {
        switch (fmt) {
        case Format::RGB_8:
            planer_count = 1;
            pixel_fmt = Format::RGB_8;
            pixel_fmt = kCVPixelFormatType_32BGRA;
            first_tex_fmt = MTLPixelFormatBGRA8Unorm;
            break;
        case Format::RGB_10:
            planer_count = 1;
            pixel_fmt = Format::RGB_10;
            pixel_fmt = kCVPixelFormatType_ARGB2101010LEPacked;
            first_tex_fmt = MTLPixelFormatRGB10A2Unorm;
            break;
        case Format::NV12_8:
            planer_count = 2;
            pixel_fmt = Format::NV12_8;
            pixel_fmt = kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange;
            first_tex_fmt = MTLPixelFormatR8Unorm;
            second_tex_fmt = MTLPixelFormatRG8Unorm;
            break;
        case Format::NV12_10:
            planer_count = 2;
            pixel_fmt = Format::NV12_10;
            pixel_fmt = kCVPixelFormatType_420YpCbCr10BiPlanarVideoRange;
            first_tex_fmt = MTLPixelFormatR16Unorm;
            second_tex_fmt = MTLPixelFormatRG16Unorm;
            break;
        default:
            break;
        }
    }
    OSType pixel_fmt;
    MTLPixelFormat first_tex_fmt;
    MTLPixelFormat second_tex_fmt;
    MTLPixelFormat third_tex_fmt;
    size_t planer_count;
    Format format;
};

class MetalHelper {
  public:
    enum HDR_FMT {
        NOT_HDR = 0,
        HDR = 1,
        HLG = 2,
        PQ = 3,
    };
    virtual ~MetalHelper();
    MetalHelper() = default;

    static MetalHelper &getSingleInstance() {
        static MetalHelper instance;
        instance.init();
        return instance;
    }

    int init();

    int createMTLTextureByCVPixelBufferRef(CVPixelBufferRef buffer,
                                           id<MTLTexture> __strong *texture,
                                           int plane,
                                           MTLPixelFormat plane_pixel_type);

    int createMTLTextureByCVPixelBufferRef(CVPixelBufferRef buffer,
                                           __strong id<MTLTexture> *tex0,
                                           __strong id<MTLTexture> *tex1,
                                           __strong id<MTLTexture> *tex2);

    int createMTLTextureRefByCVPixelBufferRef(CVPixelBufferRef buffer,
                                              CVMetalTextureRef *tex_ref0,
                                              CVMetalTextureRef *tex_ref1,
                                              CVMetalTextureRef *tex_ref2);
    int createMTLTextureRefByCVPixelBufferRef(CVPixelBufferRef buffer,
                                              CVMetalTextureRef *tex_ref,
                                              int plane,
                                              MTLPixelFormat plane_pixel_type);

    int createMTLTexture(id<MTLTexture> __strong *otexture,
                         MTLPixelFormat pixel_format, size_t width,
                         size_t height, MTLTextureUsage usage,
                         MTLStorageMode mode);

    id<MTLTexture> createMTLTexture(int width, int height,
                                    MTLPixelFormat format);

    int copy_texture(id<MTLTexture> srcs, id<MTLTexture> dsts);

    id<MTLDevice> getMTLDevice();

    id<MTLCommandQueue> getMTLCommandQueue();

    int createCVPixelBuffer(size_t width, size_t height, OSType format,
                            CVPixelBufferRef *buf_ptr);

    int createCvPixelBufferAndTexture(int width, int height, uint32_t format,
                                      CVPixelBufferRef refer_buf,
                                      CVPixelBufferRef *buf,
                                      id<MTLTexture> *tex0,
                                      id<MTLTexture> *tex1,
                                      id<MTLTexture> *tex2, bool only_tex);

    int getHdrFmt(CVPixelBufferRef buf);

    // private:
    id<MTLDevice> device_ = nil;
    id<MTLCommandQueue> command_queue_ = nil;
    id<MTLLibrary> default_library_ = nil;
    CVMetalTextureCacheRef metal_cache_ = nil;
    std::atomic<bool> init_ = {false};
}; // end class MetalHelper

BMFLITE_DEMO_NAMESPACE_END

#endif /* _BMFLITE_DEMO_METAL_HELPER_H_ */
