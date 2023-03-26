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

#include <kernel/imgproc.h>
#include <kernel/cv2/cv2_helper.h>

namespace hmp{
namespace kernel{
namespace{


inline Tensor &img_morph_cpu(cv::MorphTypes algo, 
    Tensor &dst, const Tensor &src, const Tensor &kernel, ChannelFormat cformat)
{
    auto kmat = ocv::to_cv_mat(kernel, false);
    ocv::foreach_image([&](cv::Mat &dmat, const cv::Mat &smat){
        ocv::morph(algo, smat, dmat, kmat);  
    }, cformat, dst, src);

    return dst;
}


Tensor &img_erode_cpu(Tensor &dst, const Tensor &src, const Tensor &kernel,
    ChannelFormat cformat)
{
    return img_morph_cpu(cv::MORPH_ERODE, dst, src, kernel, cformat);
}


Tensor &img_dilate_cpu(Tensor &dst, const Tensor &src, const Tensor &kernel,
    ChannelFormat cformat)
{
    return img_morph_cpu(cv::MORPH_DILATE, dst, src, kernel, cformat);
}



Tensor &img_sobel_cpu(Tensor &dst, const Tensor &src, int64_t dx, int64_t dy,
    int64_t ksize, const Scalar& scale, const Scalar& delta,
    ChannelFormat cformat)
{
    ocv::foreach_image([&](cv::Mat &dmat, const cv::Mat &smat){
        auto ddepth = ocv::to_cv_type(dst.dtype());
        cv::Sobel(smat, dmat, ddepth, dx, dy,
             ksize, scale.to<double>(), delta.to<double>());
    }, cformat, dst, src);

    return dst;
}


Tensor &img_canny_cpu(Tensor &dst, const Tensor &src, 
    const Scalar &threshold1, const Scalar &threshold2,
    int64_t aperture_size, bool l2_gradient, ChannelFormat cformat)
{
    auto stmp = src;
    auto dtmp = dst;
    if(cformat == kNCHW && src.size(1) > 1){
        stmp = src.permute({0, 2, 3, 1}).contiguous(); //to NHWC
        HMP_REQUIRE(dst.size(1) == 1, "img_canny_cpu: invalid dst shape");
        dtmp = dst.squeeze(1).unsqueeze_(-1); //to NHWC
        cformat = kNHWC;
    }

    ocv::foreach_image([&](cv::Mat &dmat, const cv::Mat &smat){
        cv::Canny(smat, dmat, threshold1.to<double>(), threshold2.to<double>(),
                  aperture_size, l2_gradient);
    }, cformat, dtmp, stmp);

    return dst;
}


Tensor &img_filter2d_cpu(Tensor &dst, const Tensor &src,
    const Tensor &kernel, const Scalar& delta, ChannelFormat cformat)
{
    auto kmat = ocv::to_cv_mat(kernel, false);
    ocv::foreach_image([&](cv::Mat &dmat, const cv::Mat &smat){
        auto ddepth = ocv::to_cv_type(dst.dtype());
        cv::filter2D(smat, dmat, ddepth, kmat,
            cv::Point(-1,-1), delta.to<double>());
    }, cformat, dst, src);

    return dst;
}


Tensor& img_gaussian_blur_cpu(Tensor &dst, const Tensor &src, int kx, int ky,
     const Scalar& sigma_x, const Scalar& sigma_y, ChannelFormat cformat)
{
    ocv::foreach_image([&](cv::Mat &dmat, const cv::Mat &smat){
        cv::GaussianBlur(smat, dmat, {kx, ky},
            sigma_x.to<double>(), sigma_y.to<double>());
    }, cformat, dst, src);

    return dst;
}



Tensor& img_bilateral_filter_cpu(Tensor &dst, const Tensor &src, int d, 
    const Scalar& sigma_color, const Scalar& sigma_space, ChannelFormat cformat)
{
    auto stmp = src;
    auto dtmp = dst;
    if(cformat == kNCHW){
        stmp = src.permute({0, 2, 3, 1}).contiguous();
        dtmp = empty_like(stmp);
    }

    ocv::foreach_image([&](cv::Mat &dmat, const cv::Mat &smat){
        cv::bilateralFilter(smat, dmat, d, 
            sigma_color.to<double>(), sigma_space.to<double>());
    }, kNHWC, dtmp, stmp);

    if(cformat == kNCHW){
        dst.copy_(dtmp.permute({0, 3, 1, 2}));
    }

    return dst;
}



Tensor &img_warp_perspective_cpu(Tensor &dst, const Tensor &src,
    const Tensor &M, ImageFilterMode mode, ChannelFormat cformat)
{
    auto m = ocv::to_cv_mat(M, false);
    ocv::foreach_image([&](cv::Mat &dmat, const cv::Mat &smat){
        auto dsize = cv::Size(dmat.cols, dmat.rows);
        auto flags = ocv::to_cv_filter_mode(mode);
        cv::warpPerspective(smat, dmat, m, dsize, flags, cv::BORDER_REPLICATE);
    }, cformat, dst, src);

    return dst;
}


HMP_DEVICE_DISPATCH(kCPU, img_erode_stub, &img_erode_cpu);
HMP_DEVICE_DISPATCH(kCPU, img_dilate_stub, &img_dilate_cpu);
HMP_DEVICE_DISPATCH(kCPU, img_sobel_stub, &img_sobel_cpu);
HMP_DEVICE_DISPATCH(kCPU, img_canny_stub, &img_canny_cpu);
HMP_DEVICE_DISPATCH(kCPU, img_filter2d_stub, &img_filter2d_cpu);
HMP_DEVICE_DISPATCH(kCPU, img_gaussian_blur_stub, &img_gaussian_blur_cpu);
HMP_DEVICE_DISPATCH(kCPU, img_bilateral_filter_stub, &img_bilateral_filter_cpu);
HMP_DEVICE_DISPATCH(kCPU, img_warp_perspective_stub, &img_warp_perspective_cpu);


}}} //namespace hmp::kernel