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

#ifndef _BMFLITE_DEMO_METAL_RENDER_MODULE_H_
#define _BMFLITE_DEMO_METAL_RENDER_MODULE_H_

#import "BmfLiteDemoModule.h"
#import "BmfLiteDemoVideoFrame.h"
#import "BmfLiteDemoMetalHelper.h"
#import "BmfLiteDemoMTKViewRender.h"
#import "BmfLiteDemoMacro.h"
#import <Metal/Metal.h>
#import <MetalKit/MetalKit.h>
#import <UIKit/UIKit.h>
#include <memory>

BMFLITE_DEMO_NAMESPACE_BEGIN

class MetalRenderModule : public Module {
  public:
    enum RenderMode {
        NV12 = 0,
        YUV420 = 1,
        RGBA = 2,
    };

    enum Fmt {
        BT709 = 0,
        BT601 = 1,
        HDR_HLG = 2,
        HDR_PQ = 3,
        P3_D65 = 4,
    };

  public:
    MetalRenderModule(MTKView *view, int render_mode = 0, int rotate = false);

    MetalRenderModule() = default;

    virtual ~MetalRenderModule();

    int process(std::shared_ptr<VideoFrame> data) override;

    void setSliderValue(float value);

    int init() override;

    int close() override;

  private:
    void checkAndUpdataLayerInfo(CVPixelBufferRef ibuffer);

  private:
    __weak MTKView *view_ = nil;
    BmfLiteViewRender *render_ = nullptr;
    RenderMode mode_ = RenderMode::NV12;
    CFStringRef color_space_ = nil;
    CAMetalLayer *layer_ = nil;
    OSType pre_fmt_;
    int rotate_ = false;
}; // end class MetalRenderModule

BMFLITE_DEMO_NAMESPACE_END

#endif /* _BMFLITE_DEMO_METAL_RENDER_MODULE_H_ */
