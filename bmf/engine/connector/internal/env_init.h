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
//
//

#ifndef BMF_ENGINE_ENV_INIT_H
#define BMF_ENGINE_ENV_INIT_H

#include <string>
#ifdef BMF_ENABLE_BREAKPAD
#include "client/linux/handler/exception_handler.h"
#endif

namespace bmf::internal {
class EnvInit {
  public:
#ifdef BMF_ENABLE_BREAKPAD
    google_breakpad::ExceptionHandler *handler;
#endif
    EnvInit();
    void ChangeDmpPath(std::string path);
};

inline EnvInit env_init;
} // namespace bmf::internal

#endif // BMF_ENGINE_ENV_INIT_H
