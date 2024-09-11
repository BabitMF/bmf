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

#ifndef _BMFLITE_DEMO_ERROR_CODE_H_
#define _BMFLITE_DEMO_ERROR_CODE_H_

#import "BmfLiteDemoMacro.h"

BMFLITE_DEMO_NAMESPACE_BEGIN

enum BmfLiteErrorCode : int {
    SUCCESS = 0,
    INVALID_PARAMETER = -1,
    INVALID_DATA = -2,
    MODULE_NOT_INIT = -3,
    MODULE_INIT_FAILED = -4,
    PROCESS_FAILED = -5,
    NO_PERMISSION = -6,
    TIMEOUT = -7,
    HANG = -8,
    INSUFFICIENT_CPU_MEMORY = -9,
    INSUFFICIENT_GPU_MEMORY = -10,
    DEVICE_NOT_SUPPORT = -11,
    CREATE_MTLTEXTURE_FAILED = -12,
    FUNCTION_NOT_IMPLEMENT = -13,
    PIPE_NO_CAPACITY = -14,
    VIDEO_DECODE_FAILED = -15,
    PARAM_PARSE_FAILED = -16,
    SURFACE_IS_NIL = -17,
    VIDEO_FRAME_IS_NIL = -18,
    METAL_CACHE_IS_NIL = -19,
    METAL_DEVICE_IS_NIL = -20,
    METAL_COMMAND_QUEUE_IS_NIL = -21,
    METAL_CREATE_TEXTURE_FAILED = -22,
    METAL_RUNTIME_INIT_FAILED = -23,
    CREATE_CVPIXELBUFFER_FAILED = -24,
};

BMFLITE_DEMO_NAMESPACE_END

#endif /* _BMFLITE_DEMO_ERROR_CODE_H_ */
