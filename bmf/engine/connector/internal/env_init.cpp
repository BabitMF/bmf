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

#include "env_init.h"
#ifdef BMF_ENABLE_GLOG
#include <glog/logging.h>
#endif

#ifdef BMF_ENABLE_BREAKPAD
#define __STDC_FORMAT_MACROS
#include "client/linux/handler/exception_handler.h"
static bool dumpCallback(const google_breakpad::MinidumpDescriptor &descriptor,
                         void *context, bool succeeded) {
    printf("Dump path: %s\n", descriptor.path());
    return succeeded;
}
#endif

namespace bmf::internal {
EnvInit::EnvInit() {
#ifdef BMF_ENABLE_GLOG
    google::InstallFailureSignalHandler();
#endif
#ifdef BMF_ENABLE_BREAKPAD
    std::string path("./");
    if (getenv("BMF_BREADKPAD_PATH"))
        path = getenv("BMF_BREADKPAD_PATH");
    static google_breakpad::MinidumpDescriptor descriptor(path);
    handler = new google_breakpad::ExceptionHandler(
        descriptor, NULL, dumpCallback, NULL, true, -1);
#endif
}

void EnvInit::ChangeDmpPath(std::string path) {
#ifdef BMF_ENABLE_BREAKPAD
    google_breakpad::MinidumpDescriptor descriptor(path);
    handler->set_minidump_descriptor(descriptor);
#endif
}
} // namespace bmf::internal