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

#ifndef _BMFLITE_DEMO_MACRO_H_
#define _BMFLITE_DEMO_MACRO_H_

#define BMFLITE_DEMO_NAMESPACE_BEGIN                                           \
    namespace bmf_lite {                                                       \
    namespace demo {

#define BMFLITE_DEMO_NAMESPACE_END                                             \
    }                                                                          \
    }

#define USE_BMFLITE_DEMO_NAMESPACE using namespace bmf_lite::demo;

BMFLITE_DEMO_NAMESPACE_BEGIN

class OnlyMovable {
  public:
    OnlyMovable(OnlyMovable &&other) = default;
    OnlyMovable &operator=(OnlyMovable &&other) = default;

    OnlyMovable(const OnlyMovable &) = delete;
    OnlyMovable &operator=(const OnlyMovable &) = delete;
};

class NotMovableOrCopyable {
  public:
    NotMovableOrCopyable(const NotMovableOrCopyable &) = delete;
    NotMovableOrCopyable &operator=(NotMovableOrCopyable &) = delete;

    NotMovableOrCopyable(NotMovableOrCopyable &&) = delete;
    NotMovableOrCopyable &operator=(NotMovableOrCopyable &&) = delete;
};

BMFLITE_DEMO_NAMESPACE_END

#endif /* _BMFLITE_DEMO_MACRO_H_ */
