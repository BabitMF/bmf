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

#include <hmp/imgproc.h>
#include <hmp/core/timer.h>
#include <hmp/core/stream.h>
#include <benchmark/benchmark.h>

#ifdef HMP_ENABLE_OPENCV
#include <opencv2/opencv.hpp>
#ifdef HMP_ENABLE_CUDA
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>
#include <cuda_runtime.h>
#endif
#endif

using namespace hmp;
// Define another benchmark

namespace {


const static int ResizeBatch = 100;


template<DeviceType device, ScalarType dtype, ImageFilterMode mode, ChannelFormat cformat>
void BM_img_resize(benchmark::State &state)
{
    auto swidth = state.range(0);
    auto sheight = state.range(1);
    auto dwidth = state.range(2);
    auto dheight = state.range(3);
    auto options = TensorOptions(device).dtype(dtype);
    SizeArray sshape, dshape;
    if(cformat == kNHWC){
        sshape = {ResizeBatch, sheight, swidth, 3};
        dshape = {ResizeBatch, dheight, dwidth, 3};
    }
    else{
        sshape = {ResizeBatch, 3, sheight, swidth};
        dshape = {ResizeBatch, 3, dheight, dwidth};
    }

    auto src = empty(sshape, options);
    auto dst = empty(dshape, options);

    auto timer = create_timer(device);
    for(auto _ : state){
        timer.start();
        benchmark::DoNotOptimize(
            img::resize(dst, src, mode, cformat));
        timer.stop();
        current_stream(device)->synchronize();

        state.SetIterationTime(timer.elapsed());
    }
}


BENCHMARK_TEMPLATE(BM_img_resize, kCPU, kUInt8, kBicubic, kNHWC)
    ->Args({1280, 720, 1920, 1080})->Unit(benchmark::kMicrosecond);
BENCHMARK_TEMPLATE(BM_img_resize, kCPU, kUInt8, kBicubic, kNCHW)
    ->Args({1280, 720, 1920, 1080})->Unit(benchmark::kMicrosecond);
BENCHMARK_TEMPLATE(BM_img_resize, kCPU, kUInt8, kBilinear, kNHWC)
    ->Args({1280, 720, 1920, 1080})->Unit(benchmark::kMicrosecond);
BENCHMARK_TEMPLATE(BM_img_resize, kCPU, kUInt8, kBilinear, kNCHW)
    ->Args({1280, 720, 1920, 1080})->Unit(benchmark::kMicrosecond);

#ifdef HMP_ENABLE_CUDA
BENCHMARK_TEMPLATE(BM_img_resize, kCUDA, kUInt8, kBicubic, kNCHW)
    ->Args({1280, 720, 1920, 1080})->Unit(benchmark::kMicrosecond);
BENCHMARK_TEMPLATE(BM_img_resize, kCUDA, kUInt8, kBicubic, kNHWC)
    ->Args({1280, 720, 1920, 1080})->Unit(benchmark::kMicrosecond);
BENCHMARK_TEMPLATE(BM_img_resize, kCUDA, kUInt8, kBilinear, kNCHW)
    ->Args({1280, 720, 1920, 1080})->Unit(benchmark::kMicrosecond);
BENCHMARK_TEMPLATE(BM_img_resize, kCUDA, kUInt8, kBilinear, kNHWC)
    ->Args({1280, 720, 1920, 1080})->Unit(benchmark::kMicrosecond);
#endif

#ifdef HMP_ENABLE_OPENCV


template<DeviceType device, int format, int mode>
void BM_cv_resize(benchmark::State &state)
{
    auto swidth = state.range(0);
    auto sheight = state.range(1);
    auto dwidth = state.range(2);
    auto dheight = state.range(3);

    auto src = cv::Mat::zeros(cv::Size(swidth, sheight), format);
    cv::Mat dst;

#ifdef HMP_ENABLE_CUDA
    //default stream: call_resize_cubic_glob
    //        other : call_resize_cubic_tex 
    auto stream = create_stream(kCUDA); 
    cv::cuda::GpuMat gpu_src;
    cv::cuda::GpuMat gpu_dst;
    auto cv_stream = cv::cuda::StreamAccessor::wrapStream((cudaStream_t)stream.handle());
    gpu_src.upload(src);
#endif

    auto timer = create_timer(device);
    for(auto _ : state){
        timer.start();
        for(int i = 0; i < ResizeBatch; ++i){
            if(device == kCPU){
                cv::resize(src, dst, cv::Size(dwidth, dheight), 0, 0, mode);
            }
            else{
#ifdef HMP_ENABLE_CUDA
                cv::cuda::resize(gpu_src, gpu_dst, cv::Size(dwidth, dheight), 0, 0, mode, cv_stream);
#endif
            }
        }
        timer.stop();
        current_stream(device)->synchronize();

        state.SetIterationTime(timer.elapsed());
    }
}


BENCHMARK_TEMPLATE(BM_cv_resize, kCPU, CV_8UC3, cv::INTER_CUBIC)
    ->Args({1280, 720, 1920, 1080})->Unit(benchmark::kMicrosecond);
BENCHMARK_TEMPLATE(BM_cv_resize, kCPU, CV_8UC3, cv::INTER_LINEAR)
    ->Args({1280, 720, 1920, 1080})->Unit(benchmark::kMicrosecond);

#ifdef HMP_ENABLE_CUDA
BENCHMARK_TEMPLATE(BM_cv_resize, kCUDA, CV_8UC3, cv::INTER_CUBIC)
    ->Args({1280, 720, 1920, 1080})->Unit(benchmark::kMicrosecond);
BENCHMARK_TEMPLATE(BM_cv_resize, kCUDA, CV_8UC3, cv::INTER_LINEAR)
    ->Args({1280, 720, 1920, 1080})->Unit(benchmark::kMicrosecond);
#endif


#endif


} //namespace