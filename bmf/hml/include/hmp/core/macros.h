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

#include <utility>
#include <stdint.h>
#include <functional>
#include <hmp/config.h>

///// Android related
#ifdef __ANDROID__
#include <android/api-level.h>
#ifdef __ANDROID_API_O_MR1__
#include <android/ndk-version.h>
#ifdef __NDK_MAJOR__
#define HMP_NDK_VERSION __NDK_MAJOR__
#else
#define HMP_NDK_VERSION 16
#endif
#else
#define HMP_NDK_VERSION 14
#endif
#endif

namespace hmp {

// generate unique name in a file
#define HMP_CONCAT_H(a, b) a##b
#define HMP_CONCAT(a, b) HMP_CONCAT_H(a, b)
#define HMP_UNIQUE_NAME(prefix) HMP_CONCAT(prefix, __LINE__)

//
#define HMP_STR_IMPL(M) #M
#define HMP_STR(M) HMP_STR_IMPL(M)

// interface export
#ifdef _WIN32
#ifdef HMP_BUILD_SHARED
#define HMP_API __declspec(dllexport)
#else // HMP_BUILD_SHARED
#define HMP_API __declspec(dllimport)
#endif
#else //_WIN32
#ifdef HMP_BUILD_SHARED
#define HMP_API __attribute__((__visibility__("default")))
#else // HMP_BUILD_SHARED
#define HMP_API
#endif //
#endif //_WIN32

#define HMP_C_API extern "C" HMP_API

////
#if defined(__CUDACC__)
#define HMP_HOST __host__
#define HMP_DEVICE __device__
#else
#define HMP_HOST
#define HMP_DEVICE
#endif
#define HMP_HOST_DEVICE HMP_HOST HMP_DEVICE

//
#if defined(__clang__)
#define __ubsan_ignore_float_divide_by_zero__                                  \
    __attribute__((no_sanitize("float-divide-by-zero")))
#define __ubsan_ignore_undefined__ __attribute__((no_sanitize("undefined")))
#define __ubsan_ignore_signed_int_overflow__                                   \
    __attribute__((no_sanitize("signed-integer-overflow")))
#else
#define __ubsan_ignore_float_divide_by_zero__
#define __ubsan_ignore_undefined__
#define __ubsan_ignore_signed_int_overflow__
#endif

#define HMP_DONT_TOUCH(VAR) __asm__ __volatile__("" ::"m"(VAR))

// version string
#define HMP_VERSION_STR()                                                      \
    HMP_STR(HMP_VERSION_MAJOR)                                                 \
    "." HMP_STR(HMP_VERSION_MINOR) "." HMP_STR(HMP_VERSION_PATCH)
#define HMP_DECLARE_TAG(VAR) extern "C" volatile void *VAR##Tag
// #define HMP_DEFINE_TAG(VAR) HMP_DECLARE_TAG(VAR) = (void*)&(VAR)
#define HMP_DEFINE_TAG(VAR)

// use in function to ref VAR and prevent compiler optimize out this tag
#define HMP_IMPORT_TAG(VAR) HMP_DONT_TOUCH(VAR##Tag)

/////
// Usage: ref HMP_REGISTER_ALLOCATOR
template <typename... Args> struct Register {
    using RegisterFunc = void (*)(Args... args);
    explicit Register(RegisterFunc func, Args... args) {
        // func(std::forward<Args>(args)...); //why failed??
        func(args...);
    }
};

/// Defer
namespace impl {

template <typename F> class Defer {
    std::function<void()> f_;

    void invoke_once() {
        if (f_) {
            f_();
            f_ = std::function<void()>();
        }
    }

  public:
    Defer() = delete;
    Defer(const Defer &) = delete;
    Defer(Defer &&other) { f_ = std::move(other.f_); }
    Defer(F &&f) : f_(std::move(f)) {}

    Defer &operator=(const Defer &) = delete;
    Defer &operator=(Defer &&other) {
        invoke_once();
        std::swap(f_, other.f_);
        return *this;
    }

    void cancel() { f_ = std::function<void()>(); }

    ~Defer() { invoke_once(); }
};

} // namespace impl
template <typename F> impl::Defer<F> defer(F &&f) {
    return impl::Defer<F>(std::move(f));
}

} // namespace hmp
