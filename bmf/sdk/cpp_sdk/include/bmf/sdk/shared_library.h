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

#include <string>
#include <stdexcept>
#include <memory>
#if defined __APPLE__ || defined __MINGW32__ || defined _WIN32
#else
#include <link.h>
#endif
#include <dlfcn.h>

namespace bmf_sdk {

class BMF_API SharedLibrary {
    std::shared_ptr<void> handler_;

  public:
    enum Flags { LAZY = RTLD_LAZY, GLOBAL = RTLD_GLOBAL };

    SharedLibrary() = default;

    SharedLibrary(const std::string &path, int flags = LAZY) {
        auto handler = dlopen(path.c_str(), flags);
        if (!handler) {
            std::string errstr = "Load library " + path + " failed, ";
            errstr += dlerror();
            throw std::runtime_error(errstr);
        }
        handler_ = std::shared_ptr<void>(handler, dlclose);
    }

    template <typename T> T symbol(const std::string &name) const {
        auto ptr = reinterpret_cast<T>(raw_symbol(name));
        if (ptr == nullptr) {
            throw std::runtime_error("Find symbol " + name + " failed");
        }
        return ptr;
    }

    bool has(const std::string &name) const {
        return raw_symbol(name) != nullptr;
    }

    bool is_open() const { return handler_ != nullptr; }

    void *raw_symbol(const std::string &name) const {
        return dlsym(handler_.get(), name.c_str());
    }

    static std::string symbol_location(const void *symbol) {
        Dl_info info;
        auto rc = dladdr(symbol, &info);
        if (rc) {
            return info.dli_fname;
        } else {
            throw std::runtime_error("symbol address not found");
        }
    }

    static std::string this_line_location() {
        return symbol_location((void *)(&this_line_location));
    }

    static const char *default_shared_dir() {
#ifdef __unix__
        return "lib";
#elif defined(__APPLE__)
        return "lib";
#elif defined(__MINGW32__) || defined(_WIN32)
        return "lib";
#else
#error "unsupported os"
#endif
    }

    static const char *default_prefix() {
#ifdef __unix__
        return "lib";
#elif defined(__APPLE__)
        return "lib";
#elif defined(__MINGW32__) || defined(_WIN32)
        return "";
#else
#error "unsupported os"
#endif
    }

    static const char *default_extension() {
#ifdef __unix__
        return ".so";
#elif defined(__APPLE__)
        return ".dylib";
#elif defined(__MINGW32__) || defined(_WIN32)
        return ".dll";
#else
#error "unsupported os"
#endif
    }
};

} // namespace bmf_sdk