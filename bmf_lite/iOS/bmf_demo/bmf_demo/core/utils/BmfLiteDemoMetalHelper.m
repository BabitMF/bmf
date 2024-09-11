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

#import "BmfLiteDemoMetalHelper.h"
#import "BmfLiteDemoFmt.h"
#import "BmfLiteDemoLog.h"
#import "BmfLiteDemoErrorCode.h"

BMFLITE_DEMO_NAMESPACE_BEGIN

int MetalHelper::init() {
    if (init_.load()) {
        return BmfLiteErrorCode::SUCCESS;
    }
    @autoreleasepool {
        device_ = MTLCreateSystemDefaultDevice();
        if (nil == device_) {
            BMFLITE_DEMO_LOGE("BmfMetalHelper", "call function MTLCreateSystemDefaultDevice failed.");
            return BmfLiteErrorCode::DEVICE_NOT_SUPPORT;
        }
        command_queue_ = [device_ newCommandQueue];
        if (nil == command_queue_) {
            BMFLITE_DEMO_LOGE("BmfMetalHelper", "create command queue failed.");
            return BmfLiteErrorCode::MODULE_INIT_FAILED;;
        }
        CVReturn ret = CVMetalTextureCacheCreate(kCFAllocatorDefault, nil, device_, nil, &metal_cache_);
        if(0 != ret) {
            BMFLITE_DEMO_LOGE("BmfMetalHelper", "create MetalTextureCache failed.");
            device_ = nil;
            command_queue_ = nil;
            return BmfLiteErrorCode::MODULE_INIT_FAILED;;
        }
        default_library_ = [device_ newDefaultLibrary];
        if (nil == default_library_) {
            BMFLITE_DEMO_LOGE("BmfMetalHelper", "create default library failed.");
            return BmfLiteErrorCode::MODULE_INIT_FAILED;;
        }
        init_ = true;
        return BmfLiteErrorCode::SUCCESS;
    }
}

int MetalHelper::createMTLTexture(id<MTLTexture> __strong *otexture, MTLPixelFormat pixel_format, size_t width, size_t height,
                      MTLTextureUsage usage, MTLStorageMode mode) {
    MTLTextureDescriptor *descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:pixel_format
                                                                                             width:width
                                                                                            height:height
                                                                                            mipmapped:NO];
    descriptor.usage = usage;
    descriptor.storageMode = mode;
    MTLCreateSystemDefaultDevice();
    *otexture = [device_ newTextureWithDescriptor: descriptor];
    if (nil == *otexture) {
        return BmfLiteErrorCode::METAL_CREATE_TEXTURE_FAILED;
    }
    return BmfLiteErrorCode::SUCCESS;
}

int MetalHelper::copy_texture(id<MTLTexture> src, id<MTLTexture> dst) {
    if (dst.width != src.width || dst.height != src.height) {
        BMFLITE_DEMO_LOGE("BmfMetalHelper", "check the dst texture width: %zu height: %zu, src width: %zu height: %zu\n", dst.width, dst.height, src.width, src.height);
        return BmfLiteErrorCode::PROCESS_FAILED;
    }
    if (dst.pixelFormat != src.pixelFormat) {
        BMFLITE_DEMO_LOGE("BmfMetalHelper","check the dst pixelFormat: src: %d  dst: %d\n", (int)src.pixelFormat, (int)dst.pixelFormat);
        return BmfLiteErrorCode::PROCESS_FAILED;
    }

    id<MTLCommandBuffer> command_buffer = [command_queue_ commandBuffer];
    id<MTLBlitCommandEncoder> blit_command_encoder = [command_buffer blitCommandEncoder];
    if (@available(iOS 13.0, *)) {
        [blit_command_encoder copyFromTexture: src toTexture: dst];
    } else {
        [blit_command_encoder copyFromTexture: src sourceSlice: 0 sourceLevel: 0 sourceOrigin: MTLOriginMake(0, 0, 0) sourceSize: MTLSizeMake(dst.width, dst.height, dst.depth)
                              toTexture: dst destinationSlice:0 destinationLevel:0 destinationOrigin: MTLOriginMake(0, 0, 0)];
    }
    [blit_command_encoder endEncoding];
    [command_buffer commit];
    [command_buffer waitUntilCompleted];

    return BmfLiteErrorCode::SUCCESS;
}

int MetalHelper::createMTLTextureByCVPixelBufferRef(CVPixelBufferRef buffer,
                                       id<MTLTexture> __strong *texture, int plane, MTLPixelFormat plane_pixel_type) {
    @autoreleasepool {
        size_t width = CVPixelBufferGetWidthOfPlane(buffer, plane);
        size_t height = CVPixelBufferGetHeightOfPlane(buffer, plane);
//        OSType pixel_format = CVPixelBufferGetPixelFormatType(buffer);
        CVMetalTextureRef mtl_texture_ref;
        CVReturn ret = CVMetalTextureCacheCreateTextureFromImage(kCFAllocatorDefault, metal_cache_,
                                                                 buffer, nil,
                                                                 plane_pixel_type, width, height, plane, &mtl_texture_ref);
        if (0 != ret) {
            BMFLITE_DEMO_LOGE("BmfMetalHelper", "call function CVMetalTextureCacheCreateTextureFromImage failed]");
            return BmfLiteErrorCode::CREATE_MTLTEXTURE_FAILED;
        }
        *texture = CVMetalTextureGetTexture(mtl_texture_ref);
        CFRelease(mtl_texture_ref);
        mtl_texture_ref = nil;
        return BmfLiteErrorCode::SUCCESS;
    }
}

int MetalHelper::createCVPixelBuffer(size_t width, size_t height, OSType format, CVPixelBufferRef* buf_ptr) {
    @autoreleasepool {
        NSDictionary *options = @{
                                  (__bridge NSString *)kCVPixelBufferIOSurfacePropertiesKey: @{},
                                  };
        CVReturn result = CVPixelBufferCreate(kCFAllocatorDefault, width, height, format, (__bridge CFDictionaryRef )options, buf_ptr);
        if (result != kCVReturnSuccess ||nil == (*buf_ptr)) {
            return BmfLiteErrorCode::CREATE_CVPIXELBUFFER_FAILED;
        }
        return BmfLiteErrorCode::SUCCESS;
    }
}

int MetalHelper::getHdrFmt(CVPixelBufferRef buf) {
    if (!buf) {
        return HDR_FMT::NOT_HDR;
    }
    CFStringRef transStr = (CFStringRef)CVBufferGetAttachment(buf, kCVImageBufferTransferFunctionKey, NULL);
    if (@available(iOS 11.0, *)) {
        if (transStr != nil) {
            if (CFStringCompare(transStr,kCVImageBufferTransferFunction_SMPTE_ST_2084_PQ,0) == kCFCompareEqualTo) {
                return HDR_FMT::PQ;
            }
            if (CFStringCompare(transStr,kCVImageBufferTransferFunction_ITU_R_2100_HLG,0) == kCFCompareEqualTo) {
                return HDR_FMT::HLG;
            }
        }
    }
    CFStringRef YCbCrMatrix = (CFStringRef)CVBufferGetAttachment(buf, kCVImageBufferYCbCrMatrixKey, NULL);
    if (YCbCrMatrix != nil && CFStringCompare(YCbCrMatrix,kCVImageBufferYCbCrMatrix_ITU_R_2020,0) == kCFCompareEqualTo) {
        return HDR_FMT::HDR;
    }
    CFStringRef colorStr = (CFStringRef)CVBufferGetAttachment(buf, kCVImageBufferColorPrimariesKey, NULL);
    if (colorStr != nil && CFStringCompare(colorStr,kCVImageBufferColorPrimaries_ITU_R_2020,0) == kCFCompareEqualTo) {
        return HDR_FMT::HDR;
    }
    return HDR_FMT::NOT_HDR;
}

id<MTLTexture> MetalHelper::createMTLTexture(int width, int height, MTLPixelFormat format) {
  @autoreleasepool {
    MTLTextureDescriptor *tex_desc =
    [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format
                                                       width:width
                                                      height:height
                                                   mipmapped:NO];
    tex_desc.usage = MTLTextureUsageShaderWrite | MTLTextureUsageShaderRead;
    id<MTLTexture> tex = [device_ newTextureWithDescriptor:tex_desc];
    return tex;
  }
}

int MetalHelper::createMTLTextureRefByCVPixelBufferRef(CVPixelBufferRef buffer, CVMetalTextureRef *tex_ref0, CVMetalTextureRef *tex_ref1, CVMetalTextureRef *tex_ref2) {
    @autoreleasepool {
        OSType format = CVPixelBufferGetPixelFormatType(buffer);
        BmfLiteDemoFmt* fmt = [[BmfLiteDemoFmt alloc] initWithCVPixelBufferFormat:format];
        if (nil == fmt) {
          return -1;
        }
        size_t plane_count = [fmt getPlaneCount];
        CVMetalTextureRef *texs[3] = {tex_ref0, tex_ref1, tex_ref0};
        for (size_t i = 0; i < plane_count; ++i) {
            int ret = 0;
            ret = createMTLTextureRefByCVPixelBufferRef(buffer, texs[i], i, [fmt getTexFormatByPlane:i]);
            if (0 != ret) {
                return -1;
            }
        }
    }
    return 0;
}

int MetalHelper::createMTLTextureRefByCVPixelBufferRef(CVPixelBufferRef buffer,
                                                       CVMetalTextureRef *tex_ref, int plane, MTLPixelFormat plane_pixel_type) {
    @autoreleasepool {
        size_t width = CVPixelBufferGetWidthOfPlane(buffer, plane);
        size_t height = CVPixelBufferGetHeightOfPlane(buffer, plane);
        CVReturn ret = CVMetalTextureCacheCreateTextureFromImage(kCFAllocatorDefault, metal_cache_,
                                                                 buffer, nil,
                                                                 plane_pixel_type, width, height, plane, tex_ref);
        if (0 != ret) {
            BMFLITE_DEMO_LOGE("BmfMetalHelper", "call function CVMetalTextureCacheCreateTextureFromImage failed]");
            return BmfLiteErrorCode::CREATE_MTLTEXTURE_FAILED;
        }
        return BmfLiteErrorCode::SUCCESS;
    }
}

int MetalHelper::createMTLTextureByCVPixelBufferRef(CVPixelBufferRef buffer, __strong id<MTLTexture> *tex0, __strong id<MTLTexture> *tex1, __strong id<MTLTexture> *tex2) {
    @autoreleasepool {
        OSType format = CVPixelBufferGetPixelFormatType(buffer);
        BmfLiteDemoFmt* fmt = [[BmfLiteDemoFmt alloc] initWithCVPixelBufferFormat:format];
        if (nil == fmt) {
          return -1;
        }
        size_t plane_count = [fmt getPlaneCount];
        id<MTLTexture> texs[3] = {nil, nil, nil};
        for (size_t i = 0; i < plane_count; ++i) {
            int ret = 0;
            ret = createMTLTextureByCVPixelBufferRef(buffer, &texs[0] + i, i, [fmt getTexFormatByPlane:i]);
            if (0 != ret) {
                texs[0] = nil;
                texs[1] = nil;
                texs[2] = nil;
                return -1;
            }
        }
        *tex0 = texs[0];
        *tex1 = texs[1];
        *tex2 = texs[2];
    }
    return 0;
}

int MetalHelper::createCvPixelBufferAndTexture(int width, int height, uint32_t format, CVPixelBufferRef refer_buf, CVPixelBufferRef* buf, id<MTLTexture>* tex0, id<MTLTexture>* tex1, id<MTLTexture>* tex2, bool only_texture) {
  @autoreleasepool{
    BmfLiteDemoFmt* fmt = [[BmfLiteDemoFmt alloc] initWithCVPixelBufferFormat:format];
    if (nil == fmt) {
      return -1;
    }
    if (!only_texture) {
      NSDictionary *options = @{
                            (__bridge NSString *)kCVPixelBufferIOSurfacePropertiesKey: @{},
                          };
      CVReturn result = CVPixelBufferCreate(kCFAllocatorDefault, width, height, format, (__bridge CFDictionaryRef )options, buf);
      if (result != kCVReturnSuccess ||nil == *buf) {
        return -1;
      }
      if (nil != refer_buf) {
        CVAttachmentMode mod = kCVAttachmentMode_ShouldPropagate;
        if (@available(iOS 15.0, *)) {
          const void* color_primaries = CVBufferCopyAttachment(refer_buf, kCVImageBufferColorPrimariesKey, &mod);
          const void* matrix = CVBufferCopyAttachment(refer_buf, kCVImageBufferYCbCrMatrixKey, &mod);
          const void* transfer_function = CVBufferCopyAttachment(refer_buf, kCVImageBufferTransferFunctionKey, &mod);
          CVBufferSetAttachment(*buf, kCVImageBufferColorPrimariesKey, color_primaries, mod);
          CVBufferSetAttachment(*buf, kCVImageBufferYCbCrMatrixKey, matrix, mod);
          CVBufferSetAttachment(*buf, kCVImageBufferTransferFunctionKey, transfer_function, mod);
        } else {
          const void* color_primaries = CVBufferGetAttachment(refer_buf, kCVImageBufferColorPrimariesKey, &mod);
          const void* matrix = CVBufferGetAttachment(refer_buf, kCVImageBufferYCbCrMatrixKey, &mod);
          const void* transfer_function = CVBufferGetAttachment(refer_buf, kCVImageBufferTransferFunctionKey, &mod);
          CVBufferSetAttachment(*buf, kCVImageBufferColorPrimariesKey, color_primaries, mod);
          CVBufferSetAttachment(*buf, kCVImageBufferYCbCrMatrixKey, matrix, mod);
          CVBufferSetAttachment(*buf, kCVImageBufferTransferFunctionKey, transfer_function, mod);
        }

      }
    }
    size_t plane_count = [fmt getPlaneCount];
    id<MTLTexture> texs[3] = {nil, nil, nil};
    for (size_t i = 0; i < plane_count; ++i) {
        int ret = 0;
      if (only_texture) {
        texs[i] = createMTLTexture([fmt getWidthByPlaneIndex:i WithOriginWidth:width], [fmt getHeightByPlaneIndex:i WithOriginHeight:height],
                                   [fmt getTexFormatByPlane:i]);
      } else {
        ret = createMTLTextureByCVPixelBufferRef(*buf,&texs[0] + i, i, [fmt getTexFormatByPlane:i]);
      }
      if (0 != ret) {
        texs[0] = nil;
        texs[1] = nil;
        texs[2] = nil;
        if (nil != *buf) {
          CVPixelBufferRelease(*buf);
          *buf = nil;
        }
        return -1;
      }
    }
      *tex0 = texs[0];
      *tex1 = texs[1];
      *tex2 = texs[2];
    return 0;
  }
}

MetalHelper::~MetalHelper() {
    if (nil != metal_cache_) {
      CVMetalTextureCacheFlush(metal_cache_, 0);
      CFRelease(metal_cache_);
        metal_cache_ = nil;
    }
    device_ = nil;
    command_queue_ = nil;
}

BMFLITE_DEMO_NAMESPACE_END
