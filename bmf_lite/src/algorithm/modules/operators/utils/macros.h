#include "common/error_code.h"

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
#define ALIGNX(x, a) (((x) + (a) - 1) / (a) * (a))
#define UPDIV(x, a) (((x) - 1) / (a) + 1)

#if __APPLE__
#define OPS_LOG_DEBUG(format, ...)                                             \
    NSLog(@"[D][bmf_lite][%s, %s, %d]" format, __FILE_NAME__, __FUNCTION__,    \
          __LINE__, ##__VA_ARGS__)
#define OPS_LOG_INFO(format, ...)                                              \
    NSLog(@"[I][bmf_lite][%s, %s, %d]" format, __FILE_NAME__, __FUNCTION__,    \
          __LINE__, ##__VA_ARGS__)
#define OPS_LOG_WARN(format, ...)                                              \
    NSLog(@"[W][bmf_lite][%s, %s, %d]" format, __FILE_NAME__, __FUNCTION__,    \
          __LINE__, ##__VA_ARGS__)
#define OPS_LOG_ERROR(format, ...)                                             \
    NSLog(@"[E][bmf_lite][%s, %s, %d]" format, __FILE_NAME__, __FUNCTION__,    \
          __LINE__, ##__VA_ARGS__)
#define OPS_LOG_FATAL(format, ...)                                             \
    NSLog(@"[F][bmf_lite][%s, %s, %d]" format, __FILE_NAME__, __FUNCTION__,    \
          __LINE__, ##__VA_ARGS__)
#elif __ANDROID__
#include <android/log.h>
#define OPS_LOG_DEBUG(format, ...)                                             \
    __android_log_print(ANDROID_LOG_DEBUG, "bmf_lite", "[%s, %s, %d]" format,  \
                        __FILE_NAME__, __FUNCTION__, __LINE__, ##__VA_ARGS__)
#define OPS_LOG_INFO(format, ...)                                              \
    __android_log_print(ANDROID_LOG_INFO, "bmf_lite", "[%s, %s, %d]" format,   \
                        __FILE_NAME__, __FUNCTION__, __LINE__, ##__VA_ARGS__)
#define OPS_LOG_WARN(format, ...)                                              \
    __android_log_print(ANDROID_LOG_WARN, "bmf_lite", "[%s, %s, %d]" format,   \
                        __FILE_NAME__, __FUNCTION__, __LINE__, ##__VA_ARGS__)
#define OPS_LOG_ERROR(format, ...)                                             \
    __android_log_print(ANDROID_LOG_ERROR, "bmf_lite", "[%s, %s, %d]" format,  \
                        __FILE_NAME__, __FUNCTION__, __LINE__, ##__VA_ARGS__)
#define OPS_LOG_FATAL(format, ...)                                             \
    __android_log_print(ANDROID_LOG_FATAL, "bmf_lite", "[%s, %s, %d]" format,  \
                        __FILE_NAME__, __FUNCTION__, __LINE__, ##__VA_ARGS__)
#else
#ifdef __GNUC__
#define OPS_LOG_DEBUG(format, ...)                                             \
    printf("[D][bmf_lite][%s, %s, %d]\"" format "\"\n", __BASE_FILE__,         \
           __FUNCTION__, __LINE__, ##__VA_ARGS__)
#define OPS_LOG_INFO(format, ...)                                              \
    printf("[I][bmf_lite][%s, %s, %d]\"" format "\"\n", __BASE_FILE__,         \
           __FUNCTION__, __LINE__, ##__VA_ARGS__)
#define OPS_LOG_WARN(format, ...)                                              \
    printf("[W][bmf_lite][%s, %s, %d]\"" format "\"\n", __BASE_FILE__,         \
           __FUNCTION__, __LINE__, ##__VA_ARGS__)
#define OPS_LOG_ERROR(format, ...)                                             \
    printf("[E][bmf_lite][%s, %s, %d]\"" format "\"\n", __BASE_FILE__,         \
           __FUNCTION__, __LINE__, ##__VA_ARGS__)
#define OPS_LOG_FATAL(format, ...)                                             \
    printf("[F][bmf_lite][%s, %s, %d]\"" format "\"\n", __BASE_FILE__,         \
           __FUNCTION__, __LINE__, ##__VA_ARGS__)
#else
#define OPS_LOG_DEBUG(format, ...)                                             \
    printf("[D][bmf_lite][%s, %s, %d]" format "\n", __FILE_NAME__,             \
           __FUNCTION__, __LINE__, ##__VA_ARGS__)
#define OPS_LOG_INFO(format, ...)                                              \
    printf("[I][bmf_lite][%s, %s, %d]" format "\n", __FILE_NAME__,             \
           __FUNCTION__, __LINE__, ##__VA_ARGS__)
#define OPS_LOG_WARN(format, ...)                                              \
    printf("[W][bmf_lite][%s, %s, %d]" format "\n", __FILE_NAME__,             \
           __FUNCTION__, __LINE__, ##__VA_ARGS__)
#define OPS_LOG_ERROR(format, ...)                                             \
    printf("[E][bmf_lite][%s, %s, %d]" format "\n", __FILE_NAME__,             \
           __FUNCTION__, __LINE__, ##__VA_ARGS__)
#define OPS_LOG_FATAL(format, ...)                                             \
    printf("[F][bmf_lite][%s, %s, %d]" format "\n", __FILE_NAME__,             \
           __FUNCTION__, __LINE__, ##__VA_ARGS__)
#endif
#endif

#define OPS_CHECK(cmd, format, ...)                                            \
    if (!(cmd)) {                                                              \
        OPS_LOG_ERROR(format, ##__VA_ARGS__);                                  \
        return BMF_LITE_OpsError;                                              \
    }

#define OPS_CHECK_OPENGL                                                       \
    {                                                                          \
        GLenum error = glGetError();                                           \
        if (GL_NO_ERROR != error) {                                            \
            OPS_LOG_ERROR("error_code: 0x%x", error);                          \
            return BMF_LITE_OpsError;                                          \
        }                                                                      \
    }