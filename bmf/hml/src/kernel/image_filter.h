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

//Results of cv::resize and cv::gpu::resize do not match #4728
//https://github.com/opencv/opencv/issues/4728

template<ImageFilterMode Mode, typename Iter, typename WType=typename Iter::scalar_type, typename OType = WType>
struct Filter;


template<typename Iter, typename WType, typename OType>
struct Filter<ImageFilterMode::Nearest, Iter, WType, OType>
{
    using itype = typename Iter::value_type;
    using wtype = WType;
    using otype = OType;

    HMP_HOST_DEVICE Filter(const Iter &src)
        : src_(src)
    {
    }

    HMP_HOST_DEVICE otype operator()(int batch, float x, float y) const
    {
        return src_.get(batch, int(truncf(x)), int(truncf(y)));
    }

    Iter src_;
};


template<typename Iter, typename WType, typename OType>
struct Filter<ImageFilterMode::Bilinear, Iter, WType, OType>
{
    using itype = typename Iter::value_type;
    using wtype = WType;
    using otype = OType;

    HMP_HOST_DEVICE Filter(const Iter &src)
        : src_(src)
    {
    }

    HMP_HOST_DEVICE otype operator()(int batch, float x, float y) const
    {
        wtype out(0);
        const int x1 = int(floorf(x));
        const int y1 = int(floorf(y));
        const int x2 = x1 + 1;
        const int y2 = y1 + 1;

        wtype src_reg = src_.get(batch, x1, y1);
        out = out + src_reg * ((x2 - x) * (y2 - y));

        src_reg = src_.get(batch, x2, y1);
        out = out + src_reg * ((x - x1) * (y2 - y));

        src_reg = src_.get(batch, x1, y2);
        out = out + src_reg * ((x2 - x) * (y - y1));

        src_reg = src_.get(batch, x2, y2);
        out = out + src_reg * ((x - x1) * (y - y1));

        if(std::is_integral<typename otype::value_type>::value){
            out = out.round();
        }

        return saturate_cast<otype>(out);
    }


    const Iter src_;
};


template<typename Iter, typename WType, typename OType>
struct Filter<ImageFilterMode::Bicubic, Iter, WType, OType>
{
    using itype = typename Iter::value_type;
    using wtype = WType;
    using otype = OType;

    HMP_HOST_DEVICE Filter(const Iter &src)
        : src_(src)
    {
    }

    HMP_HOST_DEVICE static inline void cubic_interp_coffs(Vector<float, 4> &coffs, float x)
    {
        const float A = -0.75f;
        coffs[0] = ((A * (x + 1) - 5 * A) * (x + 1) + 8 * A) * (x + 1) - 4 * A;
        coffs[1] = ((A + 2) * x - (A + 3)) * x * x + 1;
        coffs[2] = ((A + 2) * (1 - x) - (A + 3)) * (1 - x) * (1 - x) + 1;
        coffs[3] = 1.f - coffs[0] - coffs[1] - coffs[2];
    }


    HMP_HOST_DEVICE otype operator()(int batch, float x, float y) const
    {
        const int dx = floorf(x);
        const int dy = floorf(y);
        y -= dy; x -= dx;

        Vector<float, 4> xcoffs, ycoffs;
        cubic_interp_coffs(xcoffs, x);
        cubic_interp_coffs(ycoffs, y);

        wtype res(0);
        for (int j = 0; j < 4; ++j)
        {
            wtype sum(0);
            for(int i = 0; i < 4; ++i){
                sum += xcoffs[i] * src_.get(batch, dx - 1 + i, dy - 1 + j);
            }
            res += sum * ycoffs[j];
        }

        if(std::is_integral<typename otype::value_type>::value){
            res = res.round();
        }

        return saturate_cast<otype>(res);
    }

    Iter src_;
};



#define HMP_IMAGE_FILTER_DISPATCH_CASE(Mode, Iter, WType, OType, ...) \
    case (ImageFilterMode::Mode):{ \
        using filter_t = Filter<ImageFilterMode::Mode, Iter, WType, OType>; \
        return __VA_ARGS__();\
    }


#define HMP_DISPATCH_IMAGE_FILTER(expectMode, Iter, WType, OType, name, ...) [&](){ \
    switch(expectMode){ \
        HMP_IMAGE_FILTER_DISPATCH_CASE(Nearest, Iter, WType, OType, __VA_ARGS__)   \
        HMP_IMAGE_FILTER_DISPATCH_CASE(Bilinear, Iter, WType, OType, __VA_ARGS__)   \
        HMP_IMAGE_FILTER_DISPATCH_CASE(Bicubic, Iter, WType, OType, __VA_ARGS__)   \
        default: \
            HMP_REQUIRE(false, "Image filter mode {} is not support by {}", expectMode, #name); \
    } \
}()




}} //hmp::kernel