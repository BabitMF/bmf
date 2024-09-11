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

#ifndef _BMFLITE_ERROR_CODE_H_
#define _BMFLITE_ERROR_CODE_H_

#include <stdio.h>

namespace bmf_lite {

enum {
    BMF_LITE_StsOk = 0,
    BMF_LITE_StsNotProcess = -10,
    BMF_LITE_StsFuncNotImpl = -11,
    BMF_LITE_StsNotSupport = -12,
    BMF_LITE_StsNoMem = -100,
    BMF_LITE_StsBadArg = -200,
    BMF_LITE_StsTexTypeError = -201,
    BMF_LITE_NullPtr = -202,
    BMF_LITE_UnSupport = -203,
    BMF_LITE_OpenGlError = -300,
    BMF_LITE_OpenCLError = -400,
    BMF_LITE_EGLError = -500,
    BMF_LITE_OpsError = -600,
    BMF_LITE_HardwarebufferError = -700,
    BMF_LITE_JNI = -800,
    BMF_LITE_DSPError = -900,
    BMF_LITE_ByteNNError = -1000, // reserve -1000 ~ -1099
    BMF_LITE_ConcurrencyError = -1100,
    BMF_LITE_COREVIDEO_ERROR = -1200,
    BMF_LITE_FmtNoSupport = -1300,
    BMF_LITE_MetalCreateTextureFailed = -1401,
    BMF_LITE_MetalShaderExecFailed = -1402,
    BMF_LITE_CommandBufferFailed = -1403,
};

}

#endif //_BMFLITE_ERROR_CODE_H_