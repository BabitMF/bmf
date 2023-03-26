/*
 * Copyright 2023 Babit Authors
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
#include <bmf/sdk/error_define.h>

const char *BMFErrorStr(int status) {
    static char buf[256];

    switch (status) {
        case BMF_StsOk :
            return "No Error";
        case BMF_StsBackTrace :
            return "Backtrace";
        case BMF_StsError :
            return "Unspecified error";
        case BMF_StsInternal :
            return "Internal error";
        case BMF_StsNoMem :
            return "Insufficient memory";
        case BMF_StsBadArg :
            return "Bad argument";
        case BMF_StsNoConv :
            return "Iterations do not converge";
        case BMF_StsAutoTrace :
            return "Autotrace call";
        case BMF_StsBadSize :
            return "Incorrect size of input array";
        case BMF_StsNullPtr :
            return "Null pointer";
        case BMF_StsDivByZero :
            return "Division by zero occurred";
        case BMF_BadStep :
            return "Image step is wrong";
        case BMF_StsInplaceNotSupported :
            return "Inplace operation is not supported";
        case BMF_StsObjectNotFound :
            return "Requested object was not found";
        case BMF_BadDepth :
            return "Input image depth is not supported by function";
        case BMF_StsUnmatchedFormats :
            return "Formats of input arguments do not match";
        case BMF_StsUnmatchedSizes :
            return "Sizes of input arguments do not match";
        case BMF_StsOutOfRange :
            return "One of the arguments\' values is out of range";
        case BMF_StsUnsupportedFormat :
            return "Unsupported format or combination of formats";
        case BMF_BadCOI :
            return "Input COI is not supported";
        case BMF_BadNumChannels :
            return "Bad number of channels";
        case BMF_StsBadFlag :
            return "Bad flag (parameter or structure field)";
        case BMF_StsBadPoint :
            return "Bad parameter of type BMFPoint";
        case BMF_StsBadMask :
            return "Bad type of mask argument";
        case BMF_StsParseError :
            return "Parsing error";
        case BMF_StsNotImplemented :
            return "The function/feature is not implemented";
        case BMF_StsBadMemBlock :
            return "Memory block has been corrupted";
        case BMF_StsAssert :
            return "Assertion failed";
        case BMF_GpuNotSupported :
            return "No CUDA support";
        case BMF_GpuApiCallError :
            return "Gpu API call";
        case BMF_OpenGlNotSupported :
            return "No OpenGL support";
        case BMF_OpenGlApiCallError :
            return "OpenGL API call";
        case BMF_TranscodeError :
            return "BMF Transcode Error";
        case BMF_TranscodeFatalError :
            return "BMF Fatal Error During Transcode";
    };

    sprintf(buf, "Unknown %s code %d", status >= 0 ? "status" : "error", status);
    return buf;
}
