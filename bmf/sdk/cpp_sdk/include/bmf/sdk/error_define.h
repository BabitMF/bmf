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
#pragma once
#ifndef C__SDK_ERROR_DEFINE_H
#define C__SDK_ERROR_DEFINE_H

typedef int BMFStatus;
#include <stdio.h>
/** @ingroup CppMdSDK
 */
/** @see BMF::Error::Code */
enum {
    BMF_StsOk = 0,             /**< everything is ok                */
    BMF_StsBackTrace = -1,     /**< pseudo error for back trace     */
    BMF_StsError = -2,         /**< unknown /unspecified error      */
    BMF_StsInternal = -3,      /**< internal error (bad state)      */
    BMF_StsNoMem = -4,         /**< insufficient memory             */
    BMF_StsBadArg = -5,        /**< function arg/param is bad       */
    BMF_StsBadFunc = -6,       /**< unsupported function            */
    BMF_StsNoConv = -7,        /**< iter. didn't converge           */
    BMF_StsAutoTrace = -8,     /**< tracing                         */
    BMF_HeaderIsNull = -9,     /**< image header is NULL            */
    BMF_BadImageSize = -10,    /**< image size is invalid           */
    BMF_BadOffset = -11,       /**< offset is invalid               */
    BMF_BadDataPtr = -12,      /**/
    BMF_BadStep = -13,         /**< image step is wrong, this may happen for a
                                  non-continuous matrix */
    BMF_BadModelOrChSeq = -14, /**/
    BMF_BadNumChannels = -15,  /**< bad number of channels, for example, some
                                  functions accept only single channel matrices
                                  */
    BMF_BadNumChannel1U = -16, /**/
    BMF_BadDepth =
        -17, /**< input image depth is not supported by the function */
    BMF_BadAlphaChannel = -18, /**/
    BMF_BadOrder = -19,        /**< number of dimensions is out of range */
    BMF_BadOrigin = -20,       /**< incorrect input origin               */
    BMF_BadAlign = -21,        /**< incorrect input align                */
    BMF_BadCallBack = -22,     /**/
    BMF_BadTileSize = -23,     /**/
    BMF_BadCOI = -24,          /**< input COI is not supported           */
    BMF_BadROISize = -25,      /**< incorrect input roi                  */
    BMF_MaskIsTiled = -26,     /**/
    BMF_StsNullPtr = -27,      /**< null pointer */
    BMF_StsVecLengthErr = -28, /**< incorrect vector length */
    BMF_StsFilterStructContentErr =
        -29, /**< incorrect filter structure content */
    BMF_StsKernelStructContentErr =
        -30,                      /**< incorrect transform kernel content */
    BMF_StsFilterOffsetErr = -31, /**< incorrect filter offset value */
    BMF_StsTimeOut = -40,  /**< should be a hang caused timeout schedule */
    BMF_StsBadSize = -201, /**< the input/output structure size is incorrect  */
    BMF_StsDivByZero = -202, /**< division by zero */
    BMF_StsInplaceNotSupported =
        -203,                     /**< in-place operation is not supported */
    BMF_StsObjectNotFound = -204, /**< request can't be completed */
    BMF_StsUnmatchedFormats =
        -205,               /**< formats of input/output arrays differ */
    BMF_StsBadFlag = -206,  /**< flag is wrong or not supported */
    BMF_StsBadPoint = -207, /**< bad BMFPoint */
    BMF_StsBadMask = -208,  /**< bad format of mask (neither 8uC1 nor 8sC1)*/
    BMF_StsUnmatchedSizes =
        -209, /**< sizes of input/output structures do not match */
    BMF_StsUnsupportedFormat =
        -210, /**< the data format/type is not supported by the function*/
    BMF_StsOutOfRange = -211, /**< some of parameters are out of range */
    BMF_StsParseError =
        -212, /**< invalid syntax/structure of the parsed file */
    BMF_StsNotImplemented =
        -213, /**< the requested function/feature is not implemented */
    BMF_StsBadMemBlock = -214,     /**< an allocated block has been corrupted */
    BMF_StsAssert = -215,          /**< assertion failed   */
    BMF_GpuNotSupported = -216,    /**< no CUDA support    */
    BMF_GpuApiCallError = -217,    /**< GPU API call error */
    BMF_OpenGlNotSupported = -218, /**< no OpenGL support  */
    BMF_OpenGlApiCallError = -219, /**< OpenGL API call error */
    BMF_OpenCLApiCallError = -220, /**< OpenCL API call error */
    BMF_OpenCLDoubleNotSupported = -221,
    BMF_OpenCLInitError = -222, /**< OpenCL initialization error */
    BMF_OpenCLNoAMDBlasFft = -223,
    BMF_TranscodeError = -224,
    BMF_TranscodeFatalError = -225
};

const char *BMFErrorStr(int status);

#endif // C__SDK_ERROR_DEFINE_H
