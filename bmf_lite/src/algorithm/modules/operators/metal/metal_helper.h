#ifndef _BMF_ALGORITHM_MODULES_OPS_METAL_UTILS_H_
#define _BMF_ALGORITHM_MODULES_OPS_METAL_UTILS_H_

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "utils/macros.h"
#include <mutex>
#include "common/error_code.h"

namespace bmf_lite {
namespace metal {

class MetalHelper {
  private:
    MetalHelper() = default;

  public:
    static MetalHelper &instance() {
        static MetalHelper *g_utils_ins = nullptr;
        static std::once_flag s_once_flag;

        std::call_once(s_once_flag, [&]() { g_utils_ins = new MetalHelper(); });
        return *g_utils_ins;
    }

    MetalHelper(const MetalHelper &) = delete;
    MetalHelper &operator=(const MetalHelper &) = delete;

  public:
    int new_compute_pileline_state(id<MTLComputePipelineState> __strong *cps,
                                   NSString *func_name) {
        id<MTLLibrary> library = mtl_library();
        OPS_CHECK(library != nil, "library is nil");
        id<MTLFunction> func = [library newFunctionWithName:func_name];
        OPS_CHECK(func != nil, "newFunctionWithName %@ error", func_name);

        NSError *err = nil;
        *cps = [mtl_device() newComputePipelineStateWithFunction:func
                                                           error:&err];
        OPS_CHECK(err == nil && *cps != nil,
                  "newComputePipelineStateWithFunction %@ error:%@", func_name,
                  err);
        return BMF_LITE_StsOk;
    }

    int gen_tex(id<MTLTexture> __strong *texture, MTLPixelFormat pixel_format,
                size_t width, size_t height, MTLTextureUsage usage,
                MTLStorageMode storage_mode) {
        MTLTextureDescriptor *td = [MTLTextureDescriptor
            texture2DDescriptorWithPixelFormat:pixel_format
                                         width:width
                                        height:height
                                     mipmapped:NO];
        td.usage = usage;
        td.storageMode = storage_mode;

        *texture = [mtl_device() newTextureWithDescriptor:td];
        OPS_CHECK(*texture != nil, "newTextureWithDescriptor error");
        return BMF_LITE_StsOk;
    }

    id<MTLDevice> mtl_device() {
        if (device_ == nil) {
            device_ = MTLCreateSystemDefaultDevice();
        }
        return device_;
    }

    id<MTLLibrary> mtl_library() {
        if (library_ == nil) {
            NSError *err = nil;
            auto library_path =
                [NSBundle.mainBundle pathForResource:@"bmf_lite"
                                              ofType:@"metallib"];
            library_ = [mtl_device() newLibraryWithFile:library_path
                                                  error:&err];

            if (err != nil || library_ == nil) {
                OPS_LOG_ERROR(
                    "can not load metallib, library_path: %@, error: %@",
                    library_path, err);
                return nil;
            }
        }
        return library_;
    }

    bool support_non_uniform_tg() {
#if TARGET_OS_EMBEDDED
#if TARGET_OS_IPHONE
#if TARGET_OS_MAC
#if !TARGET_IPHONE_SIMULATOR
        if (@available(iOS 11.0, *)) {
            return [mtl_device()
                supportsFeatureSet:MTLFeatureSet_iOS_GPUFamily4_v1];
        } else {
            return false;
        }
#endif
#endif
#endif
#endif
        return true;
    }

  private:
    id<MTLDevice> device_ = nil;
    id<MTLLibrary> library_ = nil;

}; // class utils
} // namespace metal
} // namespace bmf_lite

#endif