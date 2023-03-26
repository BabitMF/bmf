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

#include <hmp/tensor.h>

namespace hmp{
namespace kernel{

template<typename T, int Size>
struct alignas(T) Vector
{
    using value_type = T;

    HMP_HOST_DEVICE inline Vector()
        : data{0}
    {
    }

    template<typename ...Args>
    explicit HMP_HOST_DEVICE inline Vector(Args &&...args)
        : data{static_cast<T>(args)...}
    {
    }

    template<typename U>
    HMP_HOST_DEVICE inline Vector(const Vector<U, Size> &v)
    {
        #pragma unroll
        for(int i = 0; i < Size; ++i){
            data[i] = v.data[i];
        }
    }

    HMP_HOST_DEVICE static constexpr int size()
    {
        return Size;
    }

    //
    HMP_HOST_DEVICE inline const value_type &operator[](int i) const
    {
        return data[i];
    }

    HMP_HOST_DEVICE inline value_type& operator[](int i)
    {
        return data[i];
    }

    HMP_HOST_DEVICE inline value_type dot(const Vector &v)
    {
        value_type sum = 0;
        #pragma unroll
        for(int i = 0; i < Size; ++i){
            sum += data[i] * v.data[i];
        }
        return sum;
    }

    template<typename U = value_type>
    HMP_HOST_DEVICE inline Vector<U, Size> round() const
    {
        Vector<U, Size> r;
        #pragma unroll
        for(int i = 0; i < Size; ++i){
            r.data[i] = ::round(data[i]);
        }
        return r;
    }

    //
#define DEFINE_INPLACE_BOP(op) \
    template<typename U> HMP_HOST_DEVICE inline Vector& operator op##=(const U &v) { \
        _Pragma("unroll")                       \
        for(int i = 0; i < Size; ++i){          \
            data[i] op##= v;                       \
        }                                       \
        return *this;                           \
    }                                           \
    template<typename U> HMP_HOST_DEVICE inline Vector& operator op##=(const Vector<U, Size> &v) { \
        _Pragma("unroll")                       \
        for(int i = 0; i < Size; ++i){          \
            data[i] op##=  v.data[i];              \
        }                                       \
        return *this;                           \
    }

    DEFINE_INPLACE_BOP(+)
    DEFINE_INPLACE_BOP(-)
    DEFINE_INPLACE_BOP(*)
    DEFINE_INPLACE_BOP(/)

#undef DEFINE_INPLACE_BOP

    T data[Size];
};


#define DEFINE_BOP(op) \
    template<typename T, typename U, int Size>          \
    HMP_HOST_DEVICE inline auto operator op(const Vector<T, Size> &v, const Vector<U, Size> &s) { \
        using R = decltype(T() op U());                 \
        Vector<R, Size> r;                              \
        _Pragma("unroll")                               \
        for(int i = 0; i < Size; ++i){                  \
            r.data[i] = v.data[i] op s.data[i];          \
        }                                               \
        return r;                                       \
    }                                                   \
    template<typename T, typename U, int Size>              \
    HMP_HOST_DEVICE inline auto operator op(const U& s, const Vector<T, Size> &v) {       \
        using R = decltype(T() op U());                 \
        Vector<R, Size> r;                              \
        _Pragma("unroll")                               \
        for(int i = 0; i < Size; ++i){                  \
            r.data[i] = s op v.data[i];                  \
        }                                               \
        return r;                                       \
    }                                                   \
    template<typename T, typename U, int Size>          \
    HMP_HOST_DEVICE inline auto operator op(const Vector<T, Size> &v, const U &s)  {      \
        using R = decltype(T() op U());                 \
        Vector<R, Size> r;                              \
        _Pragma("unroll")                               \
        for(int i = 0; i < Size; ++i){                  \
            r.data[i] = v.data[i] op s;                  \
        }                                               \
        return r;                                       \
    }

DEFINE_BOP(+)
DEFINE_BOP(-)
DEFINE_BOP(*)
DEFINE_BOP(/)

#undef DEFINE_BOP


template<typename T>
struct scalar_type_traits
{
    using type = T;
};

template<typename T, int Size>
struct scalar_type_traits<Vector<T, Size>>
{
    using type = T;
};


}} //namespace hmp::kernel