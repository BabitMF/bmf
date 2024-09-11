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

#ifndef _BMFLITE_DEMO_LOG_H_
#define _BMFLITE_DEMO_LOG_H_

#if defined(DEBUG) || defined(BMFLITE_DEMO_LOG_LEVEL_DEBUG)
#define LOGD
#define LOGI
#define LOGW
#define LOGE
#define LOGF

#else
#define LOGI
#define LOGW
#define LOGE
#define LOGF
#endif

#ifdef __APPLE__

#include <Foundation/Foundation.h>

#ifdef LOGD
#define BMFLITE_DEMO_LOGD(tag, format, ...)                                    \
    NSLog(@"[D][%s][%s, %s, %d]" format, tag, __FILE_NAME__, __FUNCTION__,     \
          __LINE__, ##__VA_ARGS__)
#else
#define BMFLITE_DEMO_LOGD(tag, format, ...)
#endif

#ifdef LOGI
#define BMFLITE_DEMO_LOGI(tag, format, ...)                                    \
    NSLog(@"[I][%s][%s, %s, %d]" format, tag, __FILE_NAME__, __FUNCTION__,     \
          __LINE__, ##__VA_ARGS__)
#else
#define BMFLITE_DEMO_LOGI(tag, format, ...)
#endif

#ifdef LOGW
#define BMFLITE_DEMO_LOGW(tag, format, ...)                                    \
    NSLog(@"[W][%s][%s, %s, %d]" format, tag, __FILE_NAME__, __FUNCTION__,     \
          __LINE__, ##__VA_ARGS__)
#else
#define BMFLITE_DEMO_LOGW(tag, format, ...)
#endif

#ifdef LOGE
#define BMFLITE_DEMO_LOGE(tag, format, ...)                                    \
    NSLog(@"[E][%s][%s, %s, %d]" format, tag, __FILE_NAME__, __FUNCTION__,     \
          __LINE__, ##__VA_ARGS__)
#else
#define BMFLITE_DEMO_LOGE(tag, format, ...)
#endif

#ifdef LOGF
#define BMFLITE_DEMO_LOGF(tag, format, ...)                                    \
    NSLog(@"[F][%s][%s, %s, %d]" format, tag, __FILE_NAME__, __FUNCTION__,     \
          __LINE__, ##__VA_ARGS__)
#else
#define BMFLITE_DEMO_LOGF(tag, format, ...)
#endif

#endif

#endif /* _BMFLITE_DEMO_LOG_H_ */
