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

#include "mtl_device_context.h"
#import <AVFoundation/AVFoundation.h>
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

namespace bmf_lite {

class MtlDeviceContextStruct {
public:
  id<MTLDevice> device;
};

MtlDeviceContext::MtlDeviceContext() {}

MtlDeviceContext::MtlDeviceContext(void *info) {
  impl_ = std::make_shared<MtlDeviceContextStruct>();
  impl_->device = (__bridge_transfer id<MTLDevice>)info;
}

MtlDeviceContext::~MtlDeviceContext() {}

std::shared_ptr<HWDeviceContext> MtlDeviceContext::storeCurrent() {
  return NULL;
}

int MtlDeviceContext::create_context() {
  impl_ = std::make_shared<MtlDeviceContextStruct>();
  impl_->device = MTLCreateSystemDefaultDevice();
  return 0;
}

int MtlDeviceContext::setCurrent() { return 0; }

int MtlDeviceContext::getContextInfo(void *&info) {
  info = (__bridge_retained void *)impl_->device;
  return 0;
}

}

#endif