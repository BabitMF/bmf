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

#ifndef _BMFLITE_MTL_DEVICE_CONTEXT_H_
#define _BMFLITE_MTL_DEVICE_CONTEXT_H_

#include "media/video_buffer/hardware_device_context.h"
#include <memory>
#include <stdio.h>

namespace bmf_lite {

class MtlDeviceContextStruct;
class MtlDeviceContext : public HWDeviceContext {
  public:
    std::shared_ptr<HWDeviceContext> storeCurrent();
    HWDeviceType deviceType() { return kHWDeviceTypeMTL; }
    MtlDeviceContext();
    MtlDeviceContext(void *info);
    ~MtlDeviceContext();
    int create_context();
    int setCurrent();
    int getContextInfo(void *&info);
    std::shared_ptr<MtlDeviceContextStruct> impl_;
};

} // namespace bmf_lite

#endif // _BMFLITE_MTL_DEVICE_CONTEXT_H_