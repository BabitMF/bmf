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

#import "BmfLiteDemoModule.h"
#import "BmfLiteDemoMacro.h"
#import "BmfLiteDemoLog.h"
#import "BmfLiteDemoErrorCode.h"

BMFLITE_DEMO_NAMESPACE_BEGIN

int Module::process(std::shared_ptr<VideoFrame> data) {
    BMFLITE_DEMO_LOGE("BMFModsEffect", "call basical process function, it not implemented.");
    return BmfLiteErrorCode::FUNCTION_NOT_IMPLEMENT;
}

int Module::init() {
    BMFLITE_DEMO_LOGE("BMFModsEffect", "call basical init function, it not implemented.");
    return BmfLiteErrorCode::FUNCTION_NOT_IMPLEMENT;
}

int Module::close() {
    BMFLITE_DEMO_LOGE("BMFModsEffect", "call basical init function, it not implemented.");
    return BmfLiteErrorCode::FUNCTION_NOT_IMPLEMENT;
}

int Module::forceClose() {
    BMFLITE_DEMO_LOGE("BMFModsEffect", "call basical init function, it not implemented.");
    return BmfLiteErrorCode::FUNCTION_NOT_IMPLEMENT;
}

std::shared_ptr<VideoFrame> Module::getCurrentVideoFrame() {
    BMFLITE_DEMO_LOGE("BMFModsEffect", "call basical getCurrentVideoFrame function, it not implemented.");
    return nullptr;
}

Module::~Module() {
}

BMFLITE_DEMO_NAMESPACE_END
