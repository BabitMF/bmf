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

#import "BmfLiteDemoPlayer.h"
#import "BmfLiteDemoMetalRenderModule.h"
#import "BmfLiteDemoFlowController.h"
#import "BmfLiteDemoDenoiseModule.h"
#import "BmfLiteDemoCannyModule.h"
#include <vector>
#include <memory>
#include <cassert>

USE_BMFLITE_DEMO_NAMESPACE

@implementation BmfLiteDemoPlayer
{
    std::vector<std::shared_ptr<Module>> _modules;
    std::shared_ptr<MetalRenderModule> render;
    BOOL _first;
}

- (instancetype)initWithMTKView:(MTKView *)view : (int)mode : (BmfLiteDemoAlgoType)algo_type {
    _first = true;
    render = std::make_shared<MetalRenderModule>(view, 0, mode);
    assert(render->init() == 0);
    _modules.clear();

    if (algo_type == BmfLiteDemoAlgoType::BMFLITE_DEMO_ALGO_DENOISE) {
        std::shared_ptr<Module> denoise = std::make_shared<BmfLiteDemoDenoiseModule>();
        assert(denoise->init() == 0);
        _modules.push_back(denoise);
    } else if (algo_type == BmfLiteDemoAlgoType::BMFLITE_DEMO_ALGO_CANNY) {
        std::shared_ptr<Module> canny = std::make_shared<BmfLiteDemoCannyModule>();
        assert(canny->init() == 0);
        _modules.push_back(canny);
    }

    _modules.push_back(render);
    return self;
}

- (void)setSliderValue:(float)value {
    render->setSliderValue(value);
}

- (void)consume:(CMSampleBufferRef)sampleBuffer : (BOOL)compare
{
    std::shared_ptr<VideoFrame> frame = std::make_shared<VideoFrame>();
    frame->eos_ = false;

    if (_first) {
        frame->first_ = true;
        _first = false;
    }
    CFRetain(sampleBuffer);
    frame->p_time_ = CMSampleBufferGetPresentationTimeStamp(sampleBuffer);
    CVPixelBufferRef pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer);
    frame->setCVPixelBufferRef(pixelBuffer);
    frame->sample_buffer_ref_ = sampleBuffer;
    if (compare) {
        frame->holdSource();
    }
    for (auto & it : _modules) {
        if (it != nullptr) {
            if (it->process(frame) != 0) break;;
        }
    }

}

@end
