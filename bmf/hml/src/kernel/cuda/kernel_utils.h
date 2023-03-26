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
#include <hmp/core/stream.h>
#include <hmp/cuda/macros.h>
#include <cuda_runtime.h>
#include <kernel/kernel_utils.h>
#include <hmp/cuda/device.h>

namespace hmp{
namespace kernel{
namespace cuda{

#ifndef __CUDACC_EXTENDED_LAMBDA__
#error "please compile with --expt-extended-lambda"
#endif


static inline cudaStream_t getCurrentCUDAStream()
{
    return reinterpret_cast<cudaStream_t>(
        current_stream(kCUDA).value().handle());
}

template<unsigned Batch, typename index_t, typename Func>
__global__ void elementwise_kernel(index_t N, Func f)
{
    index_t idx = blockIdx.x * blockDim.x * index_t(Batch) + threadIdx.x;
    for(unsigned i = 0; i < Batch; ++i){
        if(idx < N){
            f(idx);
        }
        idx += blockDim.x;
    }
}


template<unsigned NThread, unsigned Batch, typename Index, typename OT, typename Func>
inline void invoke_gen_kernel(const Func &f, int64_t N, OT *optr)
{
    dim3 block(NThread);
    dim3 grid(divup<int64_t>(N, NThread*Batch));
    auto stream = cuda::getCurrentCUDAStream();

    elementwise_kernel<Batch, Index><<<grid, block, 0, stream>>>(
        N, [=] HMP_HOST_DEVICE(Index idx) {
            optr[idx] = f(int64_t(idx));
        });
    HMP_CUDA_CHECK(cudaGetLastError());
}


template<unsigned NThread, unsigned Batch, typename Index, typename OT, typename OffCalc, typename Func>
inline void invoke_gen_kernel(const Func &f, OffCalc &offCalc, int64_t N, OT *optr)
{
    dim3 block(NThread);
    dim3 grid(divup<int64_t>(N, NThread*Batch));
    auto stream = cuda::getCurrentCUDAStream();

    elementwise_kernel<Batch, Index><<<grid, block, 0, stream>>>(
        N, [=] HMP_HOST_DEVICE(Index idx) {
            auto offs = offCalc.get(idx);
            optr[offs[0]] = f(int64_t(idx));
        });
    HMP_CUDA_CHECK(cudaGetLastError());
}

template<typename OT, typename Func>
void gen_kernel(Tensor &out, const Func &f)
{
    auto optr = out.data<OT>();
    auto N = out.nitems();
    
    if(out.is_contiguous()){
        invoke_gen_kernel<1024, 1, int64_t>(f, N, optr);
    }
    else{
        const int64_t *strides[] = {out.strides().data()};
        const int64_t *sizes = out.shape().data();

        if(N < std::numeric_limits<uint32_t>::max()){
            auto offsetCalc = OffsetCalculator<1, uint32_t>(out.dim(), sizes, strides);
            invoke_gen_kernel<1024, 1, uint32_t>(f, offsetCalc, N, optr);
        }
        else{
            auto offsetCalc = OffsetCalculator<1, int64_t>(out.dim(), sizes, strides);
            invoke_gen_kernel<1024, 1, int64_t>(f, offsetCalc, N, optr);
        }
    }
}

template<unsigned NThread, unsigned Batch, typename Index, typename OT, typename IT, typename Func>
inline void invoke_uop_kernel(const Func &f, int64_t N, OT *optr, const IT *iptr)
{
    HMP_REQUIRE(N >= 0, "element_kernel: Invalid N={}", N);

    dim3 block(NThread);
    dim3 grid(divup<int64_t>(N, NThread*Batch));
    auto stream = cuda::getCurrentCUDAStream();

    elementwise_kernel<Batch, Index><<<grid, block, 0, stream>>>(
        N, [=] HMP_HOST_DEVICE(Index idx) {
            optr[idx] = f(iptr[idx]);
        });

    HMP_CUDA_CHECK(cudaGetLastError());
}


template<unsigned NThread, unsigned Batch, typename Index, typename OT, typename IT, typename OffCalc, typename Func>
inline void invoke_uop_kernel(const Func &f, const OffCalc &offCalc, int64_t N, OT *optr, const IT *iptr)
{
    HMP_REQUIRE(N >= 0, "element_kernel: Invalid N={}", N);

    dim3 block(NThread);
    dim3 grid(divup<int64_t>(N, NThread*Batch));
    auto stream = cuda::getCurrentCUDAStream();

    elementwise_kernel<Batch, Index><<<grid, block, 0, stream>>>(
        N, [=] HMP_HOST_DEVICE(Index idx) {
            auto offs = offCalc.get(idx);
            optr[offs[0]] = f(iptr[offs[1]]);
        });

    HMP_CUDA_CHECK(cudaGetLastError());
}



template<typename OT, typename IT, typename Func>
void uop_kernel(Tensor &out, const Tensor &in, const Func &f)
{
    checkShape({out, in}, out.shape(), "uop_kernel");

    auto optr = out.data<OT>();
    auto iptr = in.data<IT>();
    auto N = out.nitems();

    if(out.is_contiguous() && in.is_contiguous()){
        invoke_uop_kernel<1024, 1, int64_t>(f, N, optr, iptr);
    }
    else{
        const int64_t *strides[] = {out.strides().data(), in.strides().data()};
        const int64_t *sizes = out.shape().data();

        if(N < std::numeric_limits<uint32_t>::max()){
            auto offsetCalc = OffsetCalculator<2, uint32_t>(out.dim(), sizes, strides);
            invoke_uop_kernel<1024, 1, uint32_t>(f, offsetCalc, N, optr, iptr);
        }
        else{
            auto offsetCalc = OffsetCalculator<2, int64_t>(out.dim(), sizes, strides);
            invoke_uop_kernel<1024, 1, int64_t>(f, offsetCalc, N, optr, iptr);
        }
    }
}


template<unsigned NThread, unsigned Batch, typename Index, typename OT, typename IT0, typename IT1, typename Func>
inline void invoke_bop_kernel(const Func &f, int64_t N, OT *optr, const IT0 *iptr0, const IT1 *iptr1)
{
    HMP_REQUIRE(N >= 0, "bop_kernel: Invalid N={}", N);

    dim3 block(NThread);
    dim3 grid(divup<int64_t>(N, NThread*Batch));
    auto stream = cuda::getCurrentCUDAStream();

    elementwise_kernel<Batch, Index><<<grid, block, 0, stream>>>(
        N, [=] HMP_HOST_DEVICE(Index idx){
             optr[idx] = f(iptr0[idx], iptr1[idx]); 
        });
    HMP_CUDA_CHECK(cudaGetLastError());
}


template<unsigned NThread, unsigned Batch, typename Index, typename OT, typename IT0, typename IT1, typename Func, typename OffCalc>
inline void invoke_bop_kernel(const Func &f, const OffCalc &offCalc, int64_t N, OT *optr, const IT0 *iptr0, const IT1 *iptr1)
{
    HMP_REQUIRE(N >= 0, "bop_kernel: Invalid N={}", N);

    dim3 block(NThread);
    dim3 grid(divup<int64_t>(N, NThread*Batch));
    auto stream = cuda::getCurrentCUDAStream();

    elementwise_kernel<Batch, Index><<<grid, block, 0, stream>>>(
        N, [=] HMP_HOST_DEVICE(Index idx){
            auto offs = offCalc.get(idx);
            optr[offs[0]] = f(iptr0[offs[1]], iptr1[offs[2]]);
        });
    HMP_CUDA_CHECK(cudaGetLastError());
}



template<typename OT, typename IT0, typename IT1, typename Func>
void bop_kernel(Tensor &out, const Tensor &in0, const Tensor &in1, const Func &f)
{
    checkShape({out, in0, in1}, out.shape(), "bop_kernel");

    auto optr = out.data<OT>();
    auto iptr0 = in0.data<IT0>();
    auto iptr1 = in1.data<IT1>();
    auto N = out.nitems();

    if(out.is_contiguous() && in0.is_contiguous() && in1.is_contiguous()){
        invoke_bop_kernel<1024, 1, int64_t>(f, N, optr, iptr0, iptr1);
    }
    else{
        const int64_t *strides[] = {out.strides().data(), in0.strides().data(), in1.strides().data()};
        const int64_t *sizes = out.shape().data();

        if(N < std::numeric_limits<uint32_t>::max()){
            auto offsetCalc = OffsetCalculator<3, uint32_t>(out.dim(), sizes, strides);
            invoke_bop_kernel<1024, 1, uint32_t>(f, offsetCalc, N, optr, iptr0, iptr1);
        }
        else{
            auto offsetCalc = OffsetCalculator<2, int64_t>(out.dim(), sizes, strides);
            invoke_bop_kernel<1024, 1, int64_t>(f, offsetCalc, N, optr, iptr0, iptr1);
        }
    }
}



template<typename Func, typename Index = int>
__global__ void img_elementwise_kernel(Func f, Index width, Index height)
{
    Index batch = blockIdx.z;
    auto xstart = blockIdx.x * blockDim.x + threadIdx.x;
    auto xstep = blockDim.x * gridDim.x;
    auto ystart = blockIdx.y * blockDim.y + threadIdx.y;
    auto ystep = blockDim.y * gridDim.y;

    for(Index h = ystart; h < height; h += ystep){
        for(Index w = xstart; w < width; w += xstep){
            f(batch, w, h);
        }
    }
}


template<typename Func, typename Index = int>
inline void invoke_img_elementwise_kernel(
    Func f, Index batch, Index width, Index height, int wthreads = 32, int hthreads = 32)
{
    dim3 block(wthreads, hthreads);
    dim3 grid(divup<int64_t>(width, block.x), 
              divup<int64_t>(height, block.y),
              batch);
    auto stream = cuda::getCurrentCUDAStream();
    img_elementwise_kernel<<<grid, block, 0, stream>>>(f, width, height);

    HMP_CUDA_CHECK(cudaGetLastError());
}



}}} //namespace