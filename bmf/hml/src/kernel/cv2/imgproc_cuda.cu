#include <kernel/imgproc.h>
#include <kernel/image_iter.h>
#include <kernel/cuda/kernel_utils.h>
#include <kernel/cv2/cv2_helper_cuda.h>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <kernel/imgproc.h>

namespace hmp{
namespace kernel{
namespace{

Tensor &img_pixel_max_cuda(Tensor &dst, const Tensor &src, ChannelFormat cformat)
{
    HMP_DISPATCH_IMAGE_TYPES_AND_HALF(src.scalar_type(), "img_pixel_max_cuda", [&](){
        auto channel = cformat == kNCHW ? src.size(1) : src.size(-1);
        HMP_DISPATCH_IMAGE_CHANNEL(channel, "img_pixel_max_cuda", [&](){
            using stype = Vector<scalar_t, C::size()>;
            using dtype = Vector<scalar_t, 1>;

            HMP_DISPATCH_CHANNEL_FORMAT(cformat, "img_pxiel_max_cuda", [&](){
                using SIter = ImageSeqIter<stype, FMT>;
                using DIter = ImageSeqIter<dtype, FMT>;
                auto src_iter = SIter::from_tensor(src, cformat);
                auto dst_iter = DIter::from_tensor(dst, cformat);

                cuda::invoke_img_elementwise_kernel([=] HMP_HOST_DEVICE(int batch, int w, int h) mutable {
                    auto s = src_iter.get(batch, w, h);

                    scalar_t v = numeric_limits<scalar_t>::lowest();
                    for(int i = 0; i < C::size(); ++i){
                        v = max(v, s[i]);
                    }

                    dst_iter.set(batch, w, h, dtype{v});
                }, dst_iter.batch_, dst_iter.width_, dst_iter.height_);
            });
        });

    });

    return dst;
}



Tensor &img_pixel_mean_cuda(Tensor &dst, const Tensor &src, ChannelFormat cformat)
{
    HMP_DISPATCH_IMAGE_TYPES_AND_HALF(src.scalar_type(), "img_pixel_max_cuda", [&](){
        auto channel = cformat == kNCHW ? src.size(1) : src.size(-1);
        HMP_DISPATCH_IMAGE_CHANNEL(channel, "img_pixel_max_cuda", [&](){
            using stype = Vector<scalar_t, C::size()>;
            using dtype = Vector<scalar_t, 1>;

            HMP_DISPATCH_CHANNEL_FORMAT(cformat, "img_pxiel_max_cuda", [&](){
                using SIter = ImageSeqIter<stype, FMT>;
                using DIter = ImageSeqIter<dtype, FMT>;
                auto src_iter = SIter::from_tensor(src, cformat);
                auto dst_iter = DIter::from_tensor(dst, cformat);

                cuda::invoke_img_elementwise_kernel([=] HMP_HOST_DEVICE(int batch, int w, int h) mutable {
                    auto s = src_iter.get(batch, w, h);

                    scalar_t v = 0;
                    for(int i = 0; i < C::size(); ++i){
                        v += round(s[i] * 1.f/C::size());
                    }

                    dst_iter.set(batch, w, h, dtype{v});
                }, dst_iter.batch_, dst_iter.width_, dst_iter.height_);
            });
        });

    });

    return dst;
}




inline Tensor &img_morph_cuda(cv::MorphTypes algo, 
    Tensor &dst, const Tensor &src, const Tensor &kernel_, ChannelFormat cformat)
{
    auto kernel = kernel_.cpu(); 
    auto kmat = ocv::to_cv_mat(kernel, false); //NOTE: opencv only support cpu kernel
    ocv::foreach_gpu_image([&](cv::cuda::GpuMat &dmat, const cv::cuda::GpuMat &smat){
        ocv::morph_cuda(algo, smat, dmat, kmat);  
    }, cformat, dst, src);

    return dst;
}


Tensor &img_erode_cuda(Tensor &dst, const Tensor &src, const Tensor &kernel,
    ChannelFormat cformat)
{
    return img_morph_cuda(cv::MORPH_ERODE, dst, src, kernel, cformat);
}


Tensor &img_dilate_cuda(Tensor &dst, const Tensor &src, const Tensor &kernel,
    ChannelFormat cformat)
{
    return img_morph_cuda(cv::MORPH_DILATE, dst, src, kernel, cformat);
}


Tensor &img_sobel_cuda(Tensor &dst, const Tensor &src, int64_t dx, int64_t dy,
    int64_t ksize, const Scalar& scale, const Scalar& delta,
    ChannelFormat cformat)
{
    auto stream = ocv::current_stream();
    ocv::foreach_gpu_image([&](cv::cuda::GpuMat &dmat, const cv::cuda::GpuMat &smat){
        auto filter = cv::cuda::createSobelFilter(
            smat.type(), dmat.type(), dx, dy, ksize, scale.to<double>());
        filter->apply(smat, dmat, stream);
    }, cformat, dst, src);

    //
    if(delta.to<double>() != 0){
        dst += delta;
    }

    return dst;
}


Tensor &img_canny_cuda(Tensor &dst, const Tensor &src, 
    const Scalar &low_thresh, const Scalar &high_thresh,
    int64_t aperture_size, bool l2_gradient, ChannelFormat cformat)
{
    auto cdim = cformat == kNCHW ? 1 : 3;
    auto stmp = src;
    if(src.size(cdim) > 1){
        auto shape = src.shape();
        shape[cdim] = 1;
        stmp = empty(shape, src.options());

        img_pixel_mean_cuda(stmp, src, cformat);
    }

    HMP_REQUIRE(stmp.size(cdim) == 1 && dst.size(cdim) == 1,
        "img_canny_cuda: invalid dst shape");

    auto dtmp = dst;
    if(cformat == kNHWC){
        dtmp = dst.squeeze(-1).unsqueeze_(1); //to NCHW
        stmp = stmp.squeeze(-1).unsqueeze_(1); //to NCHW
        cformat = kNCHW;
    }

    stmp = ocv::pitch_align_2d(stmp, true, cformat);
    auto stream = ocv::current_stream();

    ocv::foreach_gpu_image([&](cv::cuda::GpuMat &dmat, const cv::cuda::GpuMat &smat){
        auto canny = cv::cuda::createCannyEdgeDetector(
            low_thresh.to<double>(), high_thresh.to<double>(), aperture_size, l2_gradient);
        canny->detect(smat, dmat, stream);
    }, cformat, dtmp, stmp);


    return dst;
}


Tensor &img_filter2d_cuda(Tensor &dst, const Tensor &src,
    const Tensor &kernel, const Scalar& delta, ChannelFormat cformat)
{
    auto kmat = ocv::to_cv_mat(kernel, false);
    auto stream = ocv::current_stream();
    ocv::foreach_gpu_image([&](cv::cuda::GpuMat &dmat, const cv::cuda::GpuMat &smat){
        auto filter = cv::cuda::createLinearFilter(
            smat.type(), dmat.type(), kmat);
        filter->apply(smat, dmat, stream);
    }, cformat, dst, src);

    //
    if(delta.to<double>() != 0){
        dst += delta;
    }

    return dst;
}


Tensor& img_gaussian_blur_cuda(Tensor &dst, const Tensor &src, int kx, int ky,
     const Scalar& sigma_x, const Scalar& sigma_y, ChannelFormat cformat)
{
    auto stream = ocv::current_stream();
    ocv::foreach_gpu_image([&](cv::cuda::GpuMat &dmat, const cv::cuda::GpuMat &smat){
        auto filter = cv::cuda::createGaussianFilter(
            smat.type(), dmat.type(), {kx, ky}, sigma_x.to<double>(), sigma_y.to<double>());
        filter->apply(smat, dmat, stream);
    }, cformat, dst, src);

    return dst;
}


Tensor& img_bilateral_filter_cuda(Tensor &dst, const Tensor &src, int d, 
    const Scalar& sigma_color, const Scalar& sigma_space, ChannelFormat cformat)
{
    auto stmp = src;
    auto dtmp = dst;
    if(cformat == kNCHW){
        stmp = src.permute({0, 2, 3, 1}).contiguous();
        dtmp = empty_like(stmp);
    }

    auto stream = ocv::current_stream();
    ocv::foreach_gpu_image([&](cv::cuda::GpuMat &dmat, const cv::cuda::GpuMat &smat){
        cv::cuda::bilateralFilter(smat, dmat, d, 
            sigma_color.to<double>(), sigma_space.to<double>(), cv::BORDER_DEFAULT, stream);
    }, kNHWC, dtmp, stmp);

    if(cformat == kNCHW){
        dst.copy_(dtmp.permute({0, 3, 1, 2}));
    }

    return dst;
}


Tensor &img_warp_perspective_cuda(Tensor &dst, const Tensor &src,
    const Tensor &M, ImageFilterMode mode, ChannelFormat cformat)
{
    auto m = ocv::to_cv_mat(M, false);
    auto stream = ocv::current_stream();
    auto flags = ocv::to_cv_filter_mode(mode);
    ocv::foreach_gpu_image([&](cv::cuda::GpuMat &dmat, const cv::cuda::GpuMat &smat){
        auto dsize = cv::Size(dmat.cols, dmat.rows);
        cv::cuda::warpPerspective(smat, dmat, m, dsize, 
            flags, cv::BORDER_REPLICATE, cv::Scalar(), stream);
    }, cformat, dst, src);

    return dst;
}


HMP_DEVICE_DISPATCH(kCUDA, img_erode_stub, &img_erode_cuda);
HMP_DEVICE_DISPATCH(kCUDA, img_dilate_stub, &img_dilate_cuda);
HMP_DEVICE_DISPATCH(kCUDA, img_sobel_stub, &img_sobel_cuda);
HMP_DEVICE_DISPATCH(kCUDA, img_canny_stub, &img_canny_cuda);
HMP_DEVICE_DISPATCH(kCUDA, img_filter2d_stub, &img_filter2d_cuda);
HMP_DEVICE_DISPATCH(kCUDA, img_gaussian_blur_stub, &img_gaussian_blur_cuda);
HMP_DEVICE_DISPATCH(kCUDA, img_bilateral_filter_stub, &img_bilateral_filter_cuda);
HMP_DEVICE_DISPATCH(kCUDA, img_warp_perspective_stub, &img_warp_perspective_cuda);


}}} //namespace hmp::kernel