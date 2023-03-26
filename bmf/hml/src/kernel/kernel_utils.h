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
#include <array>
#include <hmp/core/macros.h>
#include <hmp/core/logging.h>
#include <hmp/core/limits.h>
#include <hmp/tensor.h>
#include <hmp/format.h>
#include <kernel/vector.h>

namespace hmp{
namespace kernel{


template<typename T, T...I>
struct integer_sequence{
    HMP_HOST_DEVICE integer_sequence(){}
};

template<size_t N, typename T, size_t ...I>
struct make_integer_sequence_impl{
    using type = typename make_integer_sequence_impl<N-1, T, N-1, I...>::type;
};

template<typename T, size_t ...I>
struct make_integer_sequence_impl<0, T, I...>{
    using type = integer_sequence<T, I...>;
};

template<size_t ...I>
using index_sequence = integer_sequence<size_t, I...>;


template<size_t N, typename T=size_t>
using make_integer_sequence = typename make_integer_sequence_impl<N, T>::type;

template<size_t N>
using make_index_sequence = typename make_integer_sequence_impl<N, size_t>::type;



template<typename T>
static inline T divup(T n, T d)
{
    static_assert(std::is_integral<T>::value, "need integer type");
    return (n + d - 1) / d;
}

template<typename U, typename T>
HMP_HOST_DEVICE inline U cast(const T &v)  
{
    //FIXME: differnet behaviour between cuda and cpu
    // cuda: clamp(lowest, max)
    // cuda:: static_cast<U>(v)
    return static_cast<U>(v);
}


template<typename T>
HMP_HOST_DEVICE inline T clamp(const T& v, const T& mn, const T &mx)
{
    return v > mx ? mx : (v < mn ? mn : v);
}


template<typename U, typename T>
struct SaturateCast{
    HMP_HOST_DEVICE inline static U cast(const T &v)
    {
        using V = decltype(T()*U()); //fix Half comparasion issue
        return clamp<V>(v, 
                        hmp::numeric_limits<U>::lowest(),
                        hmp::numeric_limits<U>::max());
    }
};



template<typename U, typename T, int Size>
struct SaturateCast<Vector<U, Size>, Vector<T, Size>>{
    HMP_HOST_DEVICE inline static Vector<U, Size> cast(const Vector<T, Size> &v)
    {
        Vector<U, Size> r;
        #pragma unroll
        for(int i = 0; i < Size; ++i){
            r[i] = SaturateCast<U, T>::cast(v[i]);
        }
        return r;
    }
};



template<typename U, typename T>
HMP_HOST_DEVICE inline U saturate_cast(const T &v)  
{
    return SaturateCast<U, T>::cast(v);
}



template<typename T>
struct IntDivider
{
    using DivMod = Vector<T, 2>;

    IntDivider() = default;
    IntDivider(T d) : d_(d) {}

    HMP_HOST_DEVICE inline T div(T v) const { return v / d_; }
    HMP_HOST_DEVICE inline T mod(T v) const { return v % d_; }
    HMP_HOST_DEVICE inline DivMod divmod(T v) const {
        return DivMod{v / d_, v % d_};
    }

    T d_ = 1;
};

template<>
struct IntDivider<uint32_t>
{
    using DivMod = Vector<uint32_t, 2>;

    IntDivider() = default;
    IntDivider(uint32_t d) : d_(d){
        for(shift_ = 0; shift_ < 32; shift_ ++){
            if((1u << shift_) >= d_){
                break;
            }
        }
        //
        uint64_t one = 1;
        uint64_t m = ((one<<32) * ((one << shift_) - d_)) / d_ + 1;
        HMP_REQUIRE(m <= std::numeric_limits<uint32_t>::max(), "Internal error");
        m_ = m;
    }

    HMP_HOST_DEVICE inline uint32_t div(uint32_t v) const {
#if defined(__CUDA__ARCH) 
        uint32_t t = __umulhi(v, m_);
#else
        uint32_t t = (uint64_t(v) * m_) >> 32;
#endif
        return (t+v)>>shift_;
    }

    HMP_HOST_DEVICE inline uint32_t mod(uint32_t v) const{
        return v - div(v) * d_;
    }

    HMP_HOST_DEVICE inline DivMod divmod(uint32_t v) const{
        uint32_t q = div(v);
        return DivMod{q, v - q * d_};
    }

    uint32_t d_ = 1;
    uint32_t m_ = 1;
    uint32_t shift_ = 0;
};


template<unsigned NArgs, typename index_t = uint32_t, unsigned MaxDims=8>
struct OffsetCalculator
{
    using offset_type = Vector<index_t, NArgs>;

    OffsetCalculator(unsigned ndim, const int64_t *sizes, const int64_t **strides) : ndim_(ndim)
    {
        HMP_REQUIRE(ndim <= MaxDims, "Tensor has to many dims(<{}), dim={}", MaxDims, ndim);
        for(unsigned i = 0; i < MaxDims; ++i){
            if(i < ndim){
                sizes_[i] = IntDivider<index_t>(sizes[i]);
            }
            else{
                sizes_[i] = IntDivider<index_t>(1);
            }

            for(unsigned arg = 0; arg < NArgs; arg ++){
                strides_[i][arg] = i < ndim ? strides[arg][i] : 0;
            }
        }
    }

    HMP_HOST_DEVICE inline offset_type get(index_t linear_idx) const {
        offset_type offsets;
        #pragma unroll
        for (int arg = 0; arg < NArgs; arg++) {
            offsets[arg] = 0;
        }

        #pragma unroll
        for (int dim = 0; dim < MaxDims; ++dim) {
            if (dim == ndim_) {
                break;
            }
            auto divmod = sizes_[ndim_ - dim - 1].divmod(linear_idx);
            linear_idx = divmod[0];

            #pragma unroll
            for (int arg = 0; arg < NArgs; arg++) {
                offsets[arg] += divmod[1] * strides_[ndim_ - dim - 1][arg];
            } 

        }
        return offsets;
    }

    //
    unsigned ndim_;
    IntDivider<index_t> sizes_[MaxDims];
    index_t strides_[MaxDims][NArgs];
};


inline void checkContiguous(const TensorList &tensors, const std::string &func)
{
    for(size_t i = 0; i < tensors.size(); ++i){
        HMP_REQUIRE(tensors.at(i).is_contiguous(),
            "{}: only support contiguous tensor, got non-contiguous tensor at {}",
            func, i);
    }
}


inline void checkDevice(const TensorList &tensors, const Device &device, const std::string &func)
{
    for(size_t i = 0; i < tensors.size(); ++i){
        HMP_REQUIRE(tensors.at(i).device() == device,
            "{}: expect tensor on device {}, got tensor at {} on {}",
            func, device, i, tensors.at(i).device());
    }
}


inline void checkShape(const TensorList &tensors, const SizeArray &shape, const std::string &func)
{
    for(size_t i = 0; i < tensors.size(); ++i){
        HMP_REQUIRE(tensors.at(i).shape() == shape,
            "{}: expect tensor has shape {}, got tensor at {} has {}",
            func, shape, i, tensors.at(i).shape());
    }
}


}} // hmp::kernel