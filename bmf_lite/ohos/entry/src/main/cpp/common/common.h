/*
 * Copyright (c) 2023 Huawei Device Co., Ltd.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef BMFLITE_COMMON_H
#define BMFLITE_COMMON_H

#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <EGL/eglplatform.h>
#include <GLES3/gl3.h>
#include <napi/native_api.h>

namespace bmf_lite_demo {
/**
 * Log print domain.
 */
const unsigned int LOG_PRINT_DOMAIN = 0xFF01;
constexpr char DEMO_NAME[] = "NativeRender";
} // namespace bmf_lite_demo
#endif // BMFLITE_COMMON_H
