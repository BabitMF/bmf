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

#include <stdint.h>
#include <limits>
#include <hmp/core/macros.h>
#include <hmp/core/details/half.h>

#ifdef __CUDACC__
#include <cuda_fp16.h>
#endif

namespace hmp {

// from: https://github.com/pytorch/pytorch/blob/master/c10/util/Half.h
struct alignas(2) Half {
    unsigned short x;

    struct from_bits_t {};
    HMP_HOST_DEVICE static constexpr from_bits_t from_bits() {
        return from_bits_t();
    }

// HIP wants __host__ __device__ tag, CUDA does not
#ifdef __HIP_PLATFORM_HCC__
    C10_HOST_DEVICE Half() = default;
#else
    Half() = default;
#endif

    constexpr HMP_HOST_DEVICE Half(unsigned short bits, from_bits_t)
        : x(bits){};

    inline HMP_HOST_DEVICE Half(float value) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
        x = __half_as_short(__float2half(value));
#else
        x = detail::fp16_ieee_from_fp32_value(value);
#endif
    }

    inline HMP_HOST_DEVICE operator float() const {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
        return __half2float(*reinterpret_cast<const __half *>(&x));
#else
        return detail::fp16_ieee_to_fp32_value(x);
#endif
    }

#if defined(__CUDACC__) || defined(__HIPCC__)
    inline HMP_HOST_DEVICE Half(const __half &value) {
        x = *reinterpret_cast<const unsigned short *>(&value);
    }
    inline HMP_HOST_DEVICE operator __half() const {
        return *reinterpret_cast<const __half *>(&x);
    }
#endif

}; // Half

// CUDA intrinsics

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 350)) ||                      \
    (defined(__clang__) && defined(__CUDA__))
inline __device__ Half __ldg(const Half *ptr) {
    return __ldg(reinterpret_cast<const __half *>(ptr));
}
#endif

/// Arithmetic

inline HMP_HOST_DEVICE Half operator+(const Half &a, const Half &b) {
    return static_cast<float>(a) + static_cast<float>(b);
}

inline HMP_HOST_DEVICE Half operator-(const Half &a, const Half &b) {
    return static_cast<float>(a) - static_cast<float>(b);
}

inline HMP_HOST_DEVICE Half operator*(const Half &a, const Half &b) {
    return static_cast<float>(a) * static_cast<float>(b);
}

inline HMP_HOST_DEVICE Half operator/(const Half &a, const Half &b)
    __ubsan_ignore_float_divide_by_zero__ {
    return static_cast<float>(a) / static_cast<float>(b);
}

inline HMP_HOST_DEVICE Half operator-(const Half &a) {
#if (defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530) ||                        \
    defined(__HIP_DEVICE_COMPILE__)
    return __hneg(a);
#else
    return -static_cast<float>(a);
#endif
}

inline HMP_HOST_DEVICE Half &operator+=(Half &a, const Half &b) {
    a = a + b;
    return a;
}

inline HMP_HOST_DEVICE Half &operator-=(Half &a, const Half &b) {
    a = a - b;
    return a;
}

inline HMP_HOST_DEVICE Half &operator*=(Half &a, const Half &b) {
    a = a * b;
    return a;
}

inline HMP_HOST_DEVICE Half &operator/=(Half &a, const Half &b) {
    a = a / b;
    return a;
}

/// Arithmetic with floats

inline HMP_HOST_DEVICE float operator+(Half a, float b) {
    return static_cast<float>(a) + b;
}
inline HMP_HOST_DEVICE float operator-(Half a, float b) {
    return static_cast<float>(a) - b;
}
inline HMP_HOST_DEVICE float operator*(Half a, float b) {
    return static_cast<float>(a) * b;
}
inline HMP_HOST_DEVICE float
operator/(Half a, float b) __ubsan_ignore_float_divide_by_zero__ {
    return static_cast<float>(a) / b;
}

inline HMP_HOST_DEVICE float operator+(float a, Half b) {
    return a + static_cast<float>(b);
}
inline HMP_HOST_DEVICE float operator-(float a, Half b) {
    return a - static_cast<float>(b);
}
inline HMP_HOST_DEVICE float operator*(float a, Half b) {
    return a * static_cast<float>(b);
}
inline HMP_HOST_DEVICE float
operator/(float a, Half b) __ubsan_ignore_float_divide_by_zero__ {
    return a / static_cast<float>(b);
}

inline HMP_HOST_DEVICE float &operator+=(float &a, const Half &b) {
    return a += static_cast<float>(b);
}
inline HMP_HOST_DEVICE float &operator-=(float &a, const Half &b) {
    return a -= static_cast<float>(b);
}
inline HMP_HOST_DEVICE float &operator*=(float &a, const Half &b) {
    return a *= static_cast<float>(b);
}
inline HMP_HOST_DEVICE float &operator/=(float &a, const Half &b) {
    return a /= static_cast<float>(b);
}

/// Arithmetic with doubles

inline HMP_HOST_DEVICE double operator+(Half a, double b) {
    return static_cast<double>(a) + b;
}
inline HMP_HOST_DEVICE double operator-(Half a, double b) {
    return static_cast<double>(a) - b;
}
inline HMP_HOST_DEVICE double operator*(Half a, double b) {
    return static_cast<double>(a) * b;
}
inline HMP_HOST_DEVICE double
operator/(Half a, double b) __ubsan_ignore_float_divide_by_zero__ {
    return static_cast<double>(a) / b;
}

inline HMP_HOST_DEVICE double operator+(double a, Half b) {
    return a + static_cast<double>(b);
}
inline HMP_HOST_DEVICE double operator-(double a, Half b) {
    return a - static_cast<double>(b);
}
inline HMP_HOST_DEVICE double operator*(double a, Half b) {
    return a * static_cast<double>(b);
}
inline HMP_HOST_DEVICE double
operator/(double a, Half b) __ubsan_ignore_float_divide_by_zero__ {
    return a / static_cast<double>(b);
}

/// Arithmetic with ints

inline HMP_HOST_DEVICE Half operator+(Half a, int b) {
    return a + static_cast<Half>(b);
}
inline HMP_HOST_DEVICE Half operator-(Half a, int b) {
    return a - static_cast<Half>(b);
}
inline HMP_HOST_DEVICE Half operator*(Half a, int b) {
    return a * static_cast<Half>(b);
}
inline HMP_HOST_DEVICE Half operator/(Half a, int b) {
    return a / static_cast<Half>(b);
}

inline HMP_HOST_DEVICE Half operator+(int a, Half b) {
    return static_cast<Half>(a) + b;
}
inline HMP_HOST_DEVICE Half operator-(int a, Half b) {
    return static_cast<Half>(a) - b;
}
inline HMP_HOST_DEVICE Half operator*(int a, Half b) {
    return static_cast<Half>(a) * b;
}
inline HMP_HOST_DEVICE Half operator/(int a, Half b) {
    return static_cast<Half>(a) / b;
}

//// Arithmetic with int64_t

inline HMP_HOST_DEVICE Half operator+(Half a, int64_t b) {
    return a + static_cast<Half>(b);
}
inline HMP_HOST_DEVICE Half operator-(Half a, int64_t b) {
    return a - static_cast<Half>(b);
}
inline HMP_HOST_DEVICE Half operator*(Half a, int64_t b) {
    return a * static_cast<Half>(b);
}
inline HMP_HOST_DEVICE Half operator/(Half a, int64_t b) {
    return a / static_cast<Half>(b);
}

inline HMP_HOST_DEVICE Half operator+(int64_t a, Half b) {
    return static_cast<Half>(a) + b;
}
inline HMP_HOST_DEVICE Half operator-(int64_t a, Half b) {
    return static_cast<Half>(a) - b;
}
inline HMP_HOST_DEVICE Half operator*(int64_t a, Half b) {
    return static_cast<Half>(a) * b;
}
inline HMP_HOST_DEVICE Half operator/(int64_t a, Half b) {
    return static_cast<Half>(a) / b;
}

/// NOTE: we do not define comparisons directly and instead rely on the implicit
/// conversion from c10::Half to float.
inline HMP_HOST_DEVICE bool operator>(Half a, Half b) {
    return static_cast<float>(a) > static_cast<float>(b);
}
inline HMP_HOST_DEVICE bool operator>=(Half a, Half b) {
    return static_cast<float>(a) >= static_cast<float>(b);
}
inline HMP_HOST_DEVICE bool operator<(Half a, Half b) {
    return static_cast<float>(a) < static_cast<float>(b);
}
inline HMP_HOST_DEVICE bool operator<=(Half a, Half b) {
    return static_cast<float>(a) <= static_cast<float>(b);
}

} // namespace hmp

namespace std {

template <> class numeric_limits<hmp::Half> {
  public:
    static constexpr bool is_specialized = true;
    static constexpr bool is_signed = true;
    static constexpr bool is_integer = false;
    static constexpr bool is_exact = false;
    static constexpr bool has_infinity = true;
    static constexpr bool has_quiet_NaN = true;
    static constexpr bool has_signaling_NaN = true;
    static constexpr auto has_denorm = numeric_limits<float>::has_denorm;
    static constexpr auto has_denorm_loss =
        numeric_limits<float>::has_denorm_loss;
    static constexpr auto round_style = numeric_limits<float>::round_style;
    static constexpr bool is_iec559 = true;
    static constexpr bool is_bounded = true;
    static constexpr bool is_modulo = false;
    static constexpr int digits = 11;
    static constexpr int digits10 = 3;
    static constexpr int max_digits10 = 5;
    static constexpr int radix = 2;
    static constexpr int min_exponent = -13;
    static constexpr int min_exponent10 = -4;
    static constexpr int max_exponent = 16;
    static constexpr int max_exponent10 = 4;
    static constexpr auto traps = numeric_limits<float>::traps;
    static constexpr auto tinyness_before =
        numeric_limits<float>::tinyness_before;
    static constexpr hmp::Half min() {
        return hmp::Half(0x0400, hmp::Half::from_bits());
    }
    static constexpr hmp::Half lowest() {
        return hmp::Half(0xFBFF, hmp::Half::from_bits());
    }
    static constexpr hmp::Half max() {
        return hmp::Half(0x7BFF, hmp::Half::from_bits());
    }
    static constexpr hmp::Half epsilon() {
        return hmp::Half(0x1400, hmp::Half::from_bits());
    }
    static constexpr hmp::Half round_error() {
        return hmp::Half(0x3800, hmp::Half::from_bits());
    }
    static constexpr hmp::Half infinity() {
        return hmp::Half(0x7C00, hmp::Half::from_bits());
    }
    static constexpr hmp::Half quiet_NaN() {
        return hmp::Half(0x7E00, hmp::Half::from_bits());
    }
    static constexpr hmp::Half signaling_NaN() {
        return hmp::Half(0x7D00, hmp::Half::from_bits());
    }
    static constexpr hmp::Half denorm_min() {
        return hmp::Half(0x0001, hmp::Half::from_bits());
    }
};

} // namespace std