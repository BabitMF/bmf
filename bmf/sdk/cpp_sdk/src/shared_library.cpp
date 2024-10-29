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

#include <bmf/sdk/common.h>
#include <bmf/sdk/shared_library.h>

#ifdef EMSCRIPTEN
#include <emscripten.h>
EM_JS(void, loadLibrary, (const char *name), {
    Asyncify.handleAsync(async () => {
        try {
        var str = UTF8ToString(name);
        await loadDynamicLibrary(str, {loadAsync: true, global: true, nodelete: true,fs : FS});
        }
        catch(error) {
        console.log(error);
        }
    });
});
#endif