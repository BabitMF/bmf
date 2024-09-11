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

#ifndef _BMFLITE_LOG_H_
#define _BMFLITE_LOG_H_

#if defined(DEBUG) || defined(BMFLITE_LOG_LEVEL_DEBUG)
#define LOGD
#define LOGI
#define LOGW
#define LOGE
#define LOGF

#elif BMFLITE_LOG_LEVEL_INFO
#define LOGI
#define LOGW
#define LOGE
#define LOGF

#elif BMFLITE_LOG_LEVEL_WARNING
#define LOGW
#define LOGE
#define LOGF

#elif BMFLITE_LOG_LEVEL_ERROR
#define LOGE
#define LOGF

#elif BMFLITE_LOG_LEVEL_FATAL
#define LOGF

#else
#define LOGI
#define LOGW
#define LOGE
#define LOGF

#endif

#ifdef __APPLE__
#import <Foundation/Foundation.h>

#ifdef LOGD
#define BMFLITE_LOGD(tag, format, ...)                                         \
    NSLog(@"[D][%s][%s, %s, %d]" format, tag, __FILE_NAME__, __FUNCTION__,     \
          __LINE__, ##__VA_ARGS__)
#else
#define BMFLITE_LOGD(tag, format, ...)
#endif

#ifdef LOGI
#define BMFLITE_LOGI(tag, format, ...)                                         \
    NSLog(@"[I][%s][%s, %s, %d]" format, tag, __FILE_NAME__, __FUNCTION__,     \
          __LINE__, ##__VA_ARGS__)
#else
#define BMFLITE_LOGI(tag, format, ...)
#endif

#ifdef LOGW
#define BMFLITE_LOGW(tag, format, ...)                                         \
    NSLog(@"[W][%s][%s, %s, %d]" format, tag, __FILE_NAME__, __FUNCTION__,     \
          __LINE__, ##__VA_ARGS__)
#else
#define BMFLITE_LOGW(tag, format, ...)
#endif

#ifdef LOGE
#define BMFLITE_LOGE(tag, format, ...)                                         \
    NSLog(@"[E][%s][%s, %s, %d]" format, tag, __FILE_NAME__, __FUNCTION__,     \
          __LINE__, ##__VA_ARGS__)
#else
#define BMFLITE_LOGE(tag, format, ...)
#endif

#ifdef LOGF
#define BMFLITE_LOGF(tag, format, ...)                                         \
    NSLog(@"[F][%s][%s, %s, %d]" format, tag, __FILE_NAME__, __FUNCTION__,     \
          __LINE__, ##__VA_ARGS__)
#else
#define BMFLITE_LOGF(tag, format, ...)
#endif

#elif __ANDROID__
#include <android/log.h>

#ifdef LOGD
#define BMFLITE_LOGD(tag, ...)                                                 \
    (__android_log_print(ANDROID_LOG_DEBUG, tag, __VA_ARGS__))
#else
#define BMFLITE_LOGD(tag, ...)
#endif

#ifdef LOGI
#define BMFLITE_LOGI(tag, ...)                                                 \
    (__android_log_print(ANDROID_LOG_INFO, tag, __VA_ARGS__))
#else
#define BMFLITE_LOGI(tag, ...)
#endif

#ifdef LOGW
#define BMFLITE_LOGW(tag, ...)                                                 \
    (__android_log_print(ANDROID_LOG_WARN, tag, __VA_ARGS__))
#else
#define BMFLITE_LOGW(tag, ...)
#endif

#ifdef LOGE
#define BMFLITE_LOGE(tag, ...)                                                 \
    (__android_log_print(ANDROID_LOG_ERROR, tag, __VA_ARGS__))
#else
#define BMFLITE_LOGE(tag, ...)
#endif

#ifdef LOGF
#define BMFLITE_LOGF(tag, ...)                                                 \
    (__android_log_print(ANDROID_LOG_FATAL, tag, __VA_ARGS__))
#else
#define BMFLITE_LOGF(tag, ...)
#endif

#ifdef LOGD
#define BMFLITE_VLOGD(tag, ...)                                                \
    (__android_log_vprint(ANDROID_LOG_DEBUG, tag, __VA_ARGS__))
#else
#define BMFLITE_VLOGD(tag, ...)
#endif

#ifdef LOGI
#define BMFLITE_VLOGI(tag, ...)                                                \
    (__android_log_vprint(ANDROID_LOG_INFO, tag, __VA_ARGS__))
#else
#define BMFLITE_VLOGI(tag, ...)
#endif

#ifdef LOGW
#define BMFLITE_VLOGW(tag, ...)                                                \
    (__android_log_vprint(ANDROID_LOG_WARN, tag, __VA_ARGS__))
#else
#define BMFLITE_VLOGW(tag, ...)
#endif

#ifdef LOGE
#define BMFLITE_VLOGE(tag, ...)                                                \
    (__android_log_vprint(ANDROID_LOG_ERROR, tag, __VA_ARGS__))
#else
#define BMFLITE_VLOGE(tag, ...)
#endif

#ifdef LOGF
#define BMFLITE_VLOGF(tag, ...)                                                \
    (__android_log_vprint(ANDROID_LOG_FATAL, tag, __VA_ARGS__))
#else
#define BMFLITE_VLOGF(tag, ...)
#endif

#elif __OHOS__
#include <hilog/log.h>

#ifdef LOGD
#define BMFLITE_LOGD(tag, ...)                                                 \
    ((void)OH_LOG_Print(LOG_APP, LOG_DEBUG, LOG_DOMAIN, tag, __VA_ARGS__))
#else
#define BMFLITE_LOGD(tag, ...)
#endif

#ifdef LOGI
#define BMFLITE_LOGI(tag, ...)                                                 \
    ((void)OH_LOG_Print(LOG_APP, LOG_INFO, LOG_DOMAIN, tag, __VA_ARGS__))
#else
#define BMFLITE_LOGI(tag, ...)
#endif

#ifdef LOGW
#define BMFLITE_LOGW(tag, ...)                                                 \
    ((void)OH_LOG_Print(LOG_APP, LOG_WARN, LOG_DOMAIN, tag, __VA_ARGS__))
#else
#define BMFLITE_LOGW(tag, ...)
#endif

#ifdef LOGE
#define BMFLITE_LOGE(tag, ...)                                                 \
    ((void)OH_LOG_Print(LOG_APP, LOG_ERROR, LOG_DOMAIN, tag, __VA_ARGS__))
#else
#define BMFLITE_LOGE(tag, ...)
#endif

#ifdef LOGF
#define BMFLITE_LOGF(tag, ...)                                                 \
    ((void)OH_LOG_Print(LOG_APP, LOG_FATAL, LOG_DOMAIN, tag, __VA_ARGS__))
#else
#define BMFLITE_LOGF(tag, ...)
#endif

#elif __GNUC__

#ifdef LOGD
#define BMFLITE_LOGD(tag, format, ...)                                         \
    printf("[D][%s][%s, %s, %d]\"" format "\"\n", tag, __BASE_FILE__,          \
           __FUNCTION__, __LINE__, ##__VA_ARGS__)
#else
#define BMFLITE_LOGD(tag, format, ...)
#endif

#ifdef LOGI
#define BMFLITE_LOGI(tag, format, ...)                                         \
    printf("[I][%s][%s, %s, %d]\"" format "\"\n", tag, __BASE_FILE__,          \
           __FUNCTION__, __LINE__, ##__VA_ARGS__)
#else
#define BMFLITE_LOGI(tag, format, ...)
#endif

#ifdef LOGW
#define BMFLITE_LOGW(tag, format, ...)                                         \
    printf("[W][%s][%s, %s, %d]\"" format "\"\n", tag, __BASE_FILE__,          \
           __FUNCTION__, __LINE__, ##__VA_ARGS__)
#else
#define BMFLITE_LOGW(tag, format, ...)
#endif

#ifdef LOGE
#define BMFLITE_LOGE(tag, format, ...)                                         \
    printf("[E][%s][%s, %s, %d]\"" format "\"\n", tag, __BASE_FILE__,          \
           __FUNCTION__, __LINE__, ##__VA_ARGS__)
#else
#define BMFLITE_LOGE(tag, format, ...)
#endif

#ifdef LOGF
#define BMFLITE_LOGF(tag, format, ...)                                         \
    printf("[F][%s][%s, %s, %d]\"" format "\"\n", tag, __BASE_FILE__,          \
           __FUNCTION__, __LINE__, ##__VA_ARGS__)
#else
#define BMFLITE_LOGF(tag, format, ...)
#endif

#else

#ifdef LOGD
#define BMFLITE_LOGD(tag, format, ...)                                         \
    printf("[D][%s][%s, %s, %d]" format "\n", tag, __FILE_NAME__,              \
           __FUNCTION__, __LINE__, ##__VA_ARGS__)
#else
#define BMFLITE_LOGD(tag, format, ...)
#endif

#ifdef LOGI
#define BMFLITE_LOGI(tag, format, ...)                                         \
    printf("[I][%s][%s, %s, %d]" format "\n", tag, __FILE_NAME__,              \
           __FUNCTION__, __LINE__, ##__VA_ARGS__)
#else
#define BMFLITE_LOGI(tag, format, ...)
#endif

#ifdef LOGW
#define BMFLITE_LOGW(tag, format, ...)                                         \
    printf("[W][%s][%s, %s, %d]" format "\n", tag, __FILE_NAME__,              \
           __FUNCTION__, __LINE__, ##__VA_ARGS__)
#else
#define BMFLITE_LOGW(tag, format, ...)
#endif

#ifdef LOGE
#define BMFLITE_LOGE(tag, format, ...)                                         \
    printf("[E][%s][%s, %s, %d]" format "\n", tag, __FILE_NAME__,              \
           __FUNCTION__, __LINE__, ##__VA_ARGS__)
#else
#define BMFLITE_LOGE(tag, format, ...)
#endif

#ifdef LOGF
#define BMFLITE_LOGF(tag, format, ...)                                         \
    printf("[F][%s][%s, %s, %d]" format "\n", tag, __FILE_NAME__,              \
           __FUNCTION__, __LINE__, ##__VA_ARGS__)
#else
#define BMFLITE_LOGF(tag, format, ...)
#endif

#endif

#endif