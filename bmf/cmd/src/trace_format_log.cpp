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

#include <bmf/sdk/common.h>
#include <bmf/sdk/trace.h>

USE_BMF_SDK_NS

int main(int argc, char **argv) {
#ifndef NO_TRACE
    // Perform log formatting to tracelog without including trace tool's
    // additional information e.g. on buffer capacity etc
    TraceLogger::instance()->format_logs(false);
#endif
    return 0;
}