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

#include <kernel/kernel_utils.h>
#include <kernel/parallel.h>

namespace hmp{
namespace kernel{
namespace cpu{

template<typename F>
inline void invoke_elementwise_kernel(int64_t N, const F &f)
{
    parallel_for(0, N, 0, [&](int64_t begin, int64_t end){
        for(int64_t i = begin; i < end; ++i){
            f(i);
        }
    });
}


template<typename OT, typename Func>
void gen_kernel(Tensor &out, const Func &f)
{
    auto optr = out.data<OT>();
    auto N = out.nitems();

    if(out.is_contiguous()){
        invoke_elementwise_kernel(N, [&](int64_t idx){
            optr[idx] = f(idx);
        });
    }
    else{
        const int64_t *strides[] = {out.strides().data()};
        const int64_t *sizes = out.shape().data();
        auto offsetCalc = OffsetCalculator<1, int64_t>(out.dim(), sizes, strides);

        invoke_elementwise_kernel(N, [&](int64_t idx){
            auto offs = offsetCalc.get(idx);
            optr[offs[0]] = f(idx);
        });
    }
}


template<typename OT, typename IT, typename Func>
void uop_kernel(Tensor &out, const Tensor &in, const Func &f)
{
    checkShape({out, in}, out.shape(), "cpu_uop_kernel");

    auto optr = out.data<OT>();
    auto iptr = in.data<IT>();
    auto N = out.nitems();

    if(in.is_contiguous() && out.is_contiguous()){
        invoke_elementwise_kernel(N, [&](int64_t idx){
            optr[idx] = f(iptr[idx]);
        });
    }
    else{
        const int64_t *strides[] = {out.strides().data(), in.strides().data()};
        const int64_t *sizes = out.shape().data();
        auto offsetCalc = OffsetCalculator<2, int64_t>(out.dim(), sizes, strides);

        invoke_elementwise_kernel(N, [&](int64_t idx){
            auto offs = offsetCalc.get(idx);
            optr[offs[0]] = f(iptr[offs[1]]);
        });
    }
}


template<typename OT, typename IT0, typename IT1, typename Func>
void bop_kernel(Tensor &out, const Tensor &in0, const Tensor &in1, const Func &f)
{
    checkShape({out, in0, in1}, out.shape(), "cpu_bop_kernel");

    auto optr = out.data<OT>();
    auto iptr0 = in0.data<IT0>();
    auto iptr1 = in1.data<IT1>();
    auto N = out.nitems();

    if(out.is_contiguous() && in0.is_contiguous() && in1.is_contiguous()){
        invoke_elementwise_kernel(N, [&](int64_t idx){
            optr[idx] = f(iptr0[idx], iptr1[idx]);
        });
    }
    else{
        const int64_t *strides[] = {out.strides().data(), in0.strides().data(), in1.strides().data()};
        const int64_t *sizes = out.shape().data();
        auto offsetCalc = OffsetCalculator<3, int64_t>(out.dim(), sizes, strides);

        invoke_elementwise_kernel(N, [&](int64_t idx){
            auto offs = offsetCalc.get(idx);
            optr[offs[0]] = f(iptr0[offs[1]], iptr1[offs[2]]);
        });
    }
}



template<typename Func, typename Index = int64_t>
inline void invoke_img_elementwise_kernel(Func f, Index batch, Index width, Index height)
{
    auto total_rows = height * batch;

    parallel_for(0, total_rows, 0, [&](Index begin, Index end){
        for(Index i = begin; i < end; ++i){
            auto b = i/height;
            auto h = i%height;
            for(Index w = 0; w < width; ++w){
                f(b, w, h);
            }
        }
    });
}



}}} //hmp::kernel::cpu