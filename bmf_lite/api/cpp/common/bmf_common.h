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

#ifndef _BMFLITE_COMMON_H_
#define _BMFLITE_COMMON_H_

#ifdef _WIN32
#define BMF_LITE_EXPORT __declspec(dllexport)
#elif __APPLE__
#define BMF_LITE_EXPORT
#elif __ANDROID__
#define BMF_LITE_EXPORT __attribute__((visibility("default")))
#elif __linux__
#define BMF_LITE_EXPORT __attribute__((visibility("default")))
#endif

#endif // _BMFLITE_COMMON_H_