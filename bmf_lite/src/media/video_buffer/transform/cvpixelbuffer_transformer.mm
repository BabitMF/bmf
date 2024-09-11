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

#ifdef BMF_LITE_ENABLE_METALBUFFER

#include "cvpixelbuffer_transformer.h"
#include "common/error_code.h"
#include "../metal_texture_video_buffer/metal_texture_video_buffer.h"
#include "../metal_texture_video_buffer/multi_metal_texture_video_buffer.h"
#import <AVFoundation/AVFoundation.h>
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

namespace bmf_lite {

PixelBufferAndTxtureFmt::PixelBufferAndTxtureFmt(OSType pixel_format) {
  pixel_buffer_fmt_ = pixel_format;
  support_ = true;
  switch (pixel_buffer_fmt_) {
        case kCVPixelFormatType_32BGRA:
            tex0_fmt_ = MTLPixelFormatBGRA8Unorm;
            tex1_fmt_ = MTLPixelFormatInvalid;
            tex2_fmt_ = MTLPixelFormatInvalid;
            plane_count_ = 1;
            plane_ratio_[0] = 0x114;
            plane_ratio_[1] = 0x00;
            plane_ratio_[2] = 0x00;
            break;
        case kCVPixelFormatType_ARGB2101010LEPacked:
            tex0_fmt_ = MTLPixelFormatRGB10A2Unorm;
            tex1_fmt_ = MTLPixelFormatInvalid;
            tex2_fmt_ = MTLPixelFormatInvalid;
            plane_count_ = 1;
            plane_ratio_[0] = 0x114;
            plane_ratio_[1] = 0x00;
            plane_ratio_[2] = 0x00;
            break;
        case kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange:
            tex0_fmt_ = MTLPixelFormatR8Unorm;
            tex1_fmt_ = MTLPixelFormatRG8Unorm;
            tex2_fmt_ = MTLPixelFormatInvalid;
            plane_count_ = 2;
            plane_ratio_[0] = 0x111;
            plane_ratio_[1] = 0x222;
            plane_ratio_[2] = 0x00;
            break;
        case kCVPixelFormatType_420YpCbCr10BiPlanarVideoRange:
            tex0_fmt_ = MTLPixelFormatR16Unorm;
            tex1_fmt_ = MTLPixelFormatRG16Unorm;
            tex2_fmt_ = MTLPixelFormatInvalid;
            plane_count_ = 2;
            plane_ratio_[0] = 0x111;
            plane_ratio_[1] = 0x222;
            plane_ratio_[2] = 0x00;
            break;
        case kCVPixelFormatType_420YpCbCr8BiPlanarFullRange:
            tex0_fmt_ = MTLPixelFormatR8Unorm;
            tex1_fmt_ = MTLPixelFormatRG8Unorm;
            tex2_fmt_ = MTLPixelFormatInvalid;
            plane_count_ = 2;
            plane_ratio_[0] = 0x111;
            plane_ratio_[1] = 0x222;
            plane_ratio_[2] = 0x00;
            break;
        case kCVPixelFormatType_420YpCbCr10BiPlanarFullRange:
            tex0_fmt_ = MTLPixelFormatR16Unorm;
            tex1_fmt_ = MTLPixelFormatRG16Unorm;
            tex2_fmt_ = MTLPixelFormatInvalid;
            plane_count_ = 2;
            plane_ratio_[0] = 0x111;
            plane_ratio_[1] = 0x222;
            plane_ratio_[2] = 0x00;
            break;
        case kCVPixelFormatType_420YpCbCr8Planar:
            tex0_fmt_ = MTLPixelFormatR8Unorm;
            tex1_fmt_ = MTLPixelFormatR8Unorm;
            tex2_fmt_ = MTLPixelFormatR8Unorm;
            plane_count_ = 3;
            plane_ratio_[0] = 0x111;
            plane_ratio_[1] = 0x221;
            plane_ratio_[2] = 0x221;
            break;
        case kCVPixelFormatType_420YpCbCr8PlanarFullRange:
            tex0_fmt_ = MTLPixelFormatR8Unorm;
            tex1_fmt_ = MTLPixelFormatR8Unorm;
            tex2_fmt_ = MTLPixelFormatR8Unorm;
            plane_count_ = 3;
            plane_ratio_[0] = 0x111;
            plane_ratio_[1] = 0x221;
            plane_ratio_[2] = 0x221;
            break;
        case kCVPixelFormatType_OneComponent16Half:
            tex0_fmt_ = MTLPixelFormatR16Float;
            tex1_fmt_ = MTLPixelFormatInvalid;
            tex2_fmt_ = MTLPixelFormatInvalid;
            plane_count_ = 1;
            plane_ratio_[0] = 0x114;
            plane_ratio_[1] = 0x00;
            plane_ratio_[2] = 0x00;
            break;
        default:
            tex0_fmt_ = MTLPixelFormatInvalid;
            tex1_fmt_ = MTLPixelFormatInvalid;
            tex2_fmt_ = MTLPixelFormatInvalid;
            plane_ratio_[0] = 0x00;
            plane_ratio_[1] = 0x00;
            plane_ratio_[2] = 0x00;
            support_ = false;
            break;
    }
}

MTLPixelFormat PixelBufferAndTxtureFmt::getTexFormatByPlane(int plane) {
  return (plane == 0) ? tex0_fmt_ : ((plane == 1) ? tex1_fmt_ : ((plane == 2) ? tex2_fmt_ : MTLPixelFormatInvalid));
}

int PixelBufferAndTxtureFmt::getPlaneCount() {
  return plane_count_;
}

int PixelBufferAndTxtureFmt::getWidthByPlaneIndexWithOriginWidth(int index, int width) {
  return width / (((plane_ratio_[index])>>4)&0xf);
}

int PixelBufferAndTxtureFmt::getHeightByPlaneIndexWithOriginHeight(int index, int height) {
  return height / (((plane_ratio_[index])>>8)&0xf);
}

bool PixelBufferAndTxtureFmt::support() {
    return support_;
}

class CvPixelBufferTransformerImpl {
public:
  std::shared_ptr<HWDeviceContext> device_context_;
  CVMetalTextureCacheRef metal_cache_ = nil;
  bool inited_ = false;

  ~CvPixelBufferTransformerImpl() {
    if (metal_cache_ != nil) {
      CVMetalTextureCacheFlush(metal_cache_, 0);
      CFRelease(metal_cache_);
      metal_cache_ = nil;
    }
  }
};

CvPixelBufferTransformer::CvPixelBufferTransformer() {}

int CvPixelBufferTransformer::init(HardwareDataInfo hardware_data_info_in,
    std::shared_ptr<HWDeviceContext> device_context,
    HardwareDataInfo hardware_data_info_out) {
  impl_ = std::make_shared<CvPixelBufferTransformerImpl>();
  impl_->inited_ = false;
  if (nullptr == device_context) {
    return BMF_LITE_StsBadArg;
  }
  impl_->device_context_ = device_context;

  void *info;
  device_context->getContextInfo(info);
  id<MTLDevice> device = (__bridge_transfer id<MTLDevice>)info;
  if (nil == device) {
    return BMF_LITE_StsBadArg;
  }

  if (hardware_data_info_in.mem_type == MemoryType::kCVPixelBuffer &&
      hardware_data_info_out.mem_type == MemoryType::kMultiMetalTexture) {
    CVReturn ret = CVMetalTextureCacheCreate(kCFAllocatorDefault, nil, device,
                                             nil, &(impl_->metal_cache_));
    if (ret == kCVReturnSuccess) {
      impl_->inited_ = true;
      return BMF_LITE_StsOk;
    }                                     
  }
  return BMF_LITE_StsBadArg;
}


int CvPixelBufferTransformer::trans(std::shared_ptr<VideoBuffer> in_video_buffer,
    std::shared_ptr<VideoBuffer> &output_video_buffer) {
  if (impl_->inited_ && in_video_buffer->memoryType() == MemoryType::kCVPixelBuffer) {
    CVPixelBufferRef ibuffer = (CVPixelBufferRef)in_video_buffer->data();
    int width = CVPixelBufferGetWidth(ibuffer);
    int height = CVPixelBufferGetHeight(ibuffer);
    OSType fmt = CVPixelBufferGetPixelFormatType(ibuffer);
    PixelBufferAndTxtureFmt fmt_mp(fmt);
    if (!fmt_mp.support()) {
      return BMF_LITE_FmtNoSupport;
    }
    size_t plane_count = fmt_mp.getPlaneCount();
    CVReturn ret = kCVReturnSuccess;
    struct VideoTextureList *texture_list = new VideoTextureList();
    texture_list->num = plane_count;
    texture_list->texture_list = (VideoBuffer **)malloc(sizeof(VideoBuffer *) * texture_list->num);

    for (size_t i = 0; i < plane_count; ++i) {
      CVMetalTextureRef tex_ref;
      MTLPixelFormat mtl_fmt = fmt_mp.getTexFormatByPlane(i);
      int plane_width = fmt_mp.getWidthByPlaneIndexWithOriginWidth(i, width);
      int plane_height = fmt_mp.getHeightByPlaneIndexWithOriginHeight(i, height);
      ret = CVMetalTextureCacheCreateTextureFromImage(kCFAllocatorDefault, impl_->metal_cache_, ibuffer, nil,
                                                               mtl_fmt, plane_width, plane_height, i, &tex_ref);
      if (kCVReturnSuccess != ret) {
        for (int j = 0; j < plane_count; ++j) {
          if (nullptr != texture_list->texture_list[i]) {
            delete texture_list->texture_list[i];
            texture_list->texture_list[i] = nullptr;
          }
        }
        if (texture_list != nullptr) {
          delete texture_list;
          texture_list = nullptr;
        }
      }
      HardwareDataInfo temp_data_info;
      // todo fmt
      temp_data_info.internal_format = mtl_fmt;
      id<MTLTexture> tex = CVMetalTextureGetTexture(tex_ref);
      void *tex_ptr = (__bridge_retained void *)tex;
      MetalTextureVideoBuffer *metal_texture_video_buffer = new MetalTextureVideoBuffer(tex_ptr, plane_width, plane_height, temp_data_info,
                                    impl_->device_context_);
      texture_list->texture_list[i] = metal_texture_video_buffer;

      CFRelease(tex_ref);
    }
    HardwareDataInfo data_info;
    data_info.mem_type = MemoryType::kMultiMetalTexture;
    // todo fmt mp
    data_info.internal_format = BMF_LITE_MTL_NV12;
    output_video_buffer = std::make_shared<MultiMetalTextureVideoBuffer>(texture_list, width, height, data_info, impl_->device_context_);
    return BMF_LITE_StsOk;
  }
  return BMF_LITE_StsNotProcess;
}

CvPixelBufferTransformer::~CvPixelBufferTransformer(){}

}

#endif
