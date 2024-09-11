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

#import "BmfLiteDemoVodPlayer.h"
#import "BmfLiteDemoVideoReaderModule.h"
#import "BmfLiteDemoAudioEngine.h"
#import "BmfLiteDemoMetalRenderModule.h"
#import "BmfLiteDemoSrModule.h"
#import "BmfLiteDemoFlowController.h"
#include <vector>
#include <memory>
#include <cassert>

USE_BMFLITE_DEMO_NAMESPACE

@implementation BmfLiteDemoVodPlayer
{
    std::shared_ptr<VideoReaderModule> _reader;
    BmfLiteDemoAudioEngine *_audio_engine;
    std::vector<std::shared_ptr<Module>> _modules;
    std::shared_ptr<MetalRenderModule> render;
    double _fps;
    bool _force_stop;
    bool _compare;
}

- (instancetype)initWithMTKView:(MTKView *)view AndVideoPath:(NSString *)path WhetherPlayAudio:(BOOL)play_audio Compare:(BOOL)compare_source {
    _force_stop = false;
    _compare = compare_source;
    _reader = std::make_shared<VideoReaderModule>(path);
    assert(_reader->init() == 0);
    _fps = _reader->getFps();

    NSURL *url = [NSURL fileURLWithPath:path];
    if (play_audio) {
//        _audio_engine = [[BmfLiteDemoAudioEngine alloc]initWithNSURL:url VideoFps:_fps];
    }
    render = std::make_shared<MetalRenderModule>(view);
    assert(render->init() == 0);

    std::shared_ptr<Module> sr = std::make_shared<BmfLiteDemoSrModule>();
    assert(sr->init() == 0);
    _modules.clear();

    _modules.push_back(sr);
    _modules.push_back(render);

    return self;
}

- (void)setSliderValue:(float)value {
    render->setSliderValue(value);
}

- (void)run {
    bool first = true;
    FlowController flow_contoller(_fps);
    while (true) {
        if (_force_stop) {
            break;
        }
        if (first) {
            if (_audio_engine != nil) {
                [_audio_engine togglePlay];
            }
            first = false;
        }

        if (0 != _reader->process(nullptr)) {
        break;
        }
        std::shared_ptr<VideoFrame> frame = _reader->getCurrentVideoFrame();
        if (nullptr == frame) {
            break;
        }
        if (_compare) {
            frame->holdSource();
        }
        if (flow_contoller.frameIsDelay(frame->p_time_)) {
            continue;
        }

        for (auto & it : _modules) {
            it->process(frame);
        }
        flow_contoller.controlByVideoPts(frame->p_time_);

        if (frame->eos_) {
            break;
        }
    }
    if (_audio_engine != nil) {
        [_audio_engine togglePlay];
    }
}

- (void)stop {
    _force_stop = true;
}

@end


