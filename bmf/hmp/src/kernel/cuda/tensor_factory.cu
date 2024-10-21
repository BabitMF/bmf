
#include <kernel/tensor_factory.h>
#include <kernel/cuda/kernel_utils.h>

namespace hmp{
namespace kernel{
namespace{


Tensor &fill_cuda_impl(Tensor &self, const Scalar &value)
{
    HMP_DISPATCH_ALL_TYPES_AND_HALF(self.scalar_type(), "fill_cuda", [&](){
        auto v = value.to<scalar_t>();

        cuda::gen_kernel<scalar_t>(self,
             [v]HMP_HOST_DEVICE(int64_t){
                return v;
             });

        HMP_CUDA_CHECK(cudaGetLastError());
    });

    return self;
}


Tensor &arange_cuda_impl(Tensor &self, int64_t start, int64_t end, int64_t step)
{
    HMP_DISPATCH_ALL_TYPES(self.scalar_type(), "arange_cuda", [&](){
        cuda::gen_kernel<scalar_t>(self,
             [=]HMP_HOST_DEVICE(int64_t idx){
                return start + idx * step;
             });

        HMP_CUDA_CHECK(cudaGetLastError());
    });

    return self;
}


Tensor &copy_cuda_impl(Tensor &self, const Tensor &other)
{
    if(self.device_type() == kCUDA && other.device_type() == kCUDA){
        HMP_REQUIRE(self.device_index() == other.device_index(),
                "copy_cuda_impl: do not support peer to peer copy");

        HMP_DISPATCH_ALL_TYPES_AND_HALF(self.scalar_type(), "copy_cuda_impl", [&](){
            using oscalar_t = scalar_t;
            HMP_DISPATCH_ALL_TYPES_AND_HALF(other.scalar_type(), "copy_cuda_impl", [&](){
                using iscalar_t = scalar_t;
                cuda::uop_kernel<oscalar_t, iscalar_t>(self, other, 
                    [=]HMP_HOST_DEVICE(iscalar_t v){
                        return cast<oscalar_t>(v);
                });

            });
        });
    }
    else{
        auto dtmp = self.is_contiguous() ? self : empty_like(self, self.options());
        auto stmp = other.is_contiguous() ? other : other.contiguous();

        HMP_DISPATCH_ALL_TYPES_AND_HALF(self.scalar_type(), "copy_cuda_impl", [&](){
            auto dst = dtmp.data<scalar_t>();
            auto src = stmp.data<scalar_t>();
            auto nbytes = dtmp.nbytes();
            auto stream = cuda::getCurrentCUDAStream();

            if(self.device_type() == kCUDA && other.device_type() == kCPU){
                HMP_CUDA_CHECK(
                    cudaMemcpyAsync(dst, src, nbytes, cudaMemcpyHostToDevice, stream));
            }
            else if(self.device_type() == kCPU && other.device_type() == kCUDA){
                HMP_CUDA_CHECK(
                    cudaMemcpyAsync(dst, src, nbytes, cudaMemcpyDeviceToHost, stream));
            }
            else{
                HMP_REQUIRE(false, "copy_cuda_impl: internal error");
            }
        });

        if(!self.is_contiguous()){
            hmp::copy(self, dtmp);
        }
    }

    return self;
}



HMP_DEVICE_DISPATCH(kCUDA, fill_stub, &fill_cuda_impl)
HMP_DEVICE_DISPATCH(kCUDA, copy_stub, &copy_cuda_impl)
HMP_DEVICE_DISPATCH(kCUDA, arange_stub, &arange_cuda_impl)


}}} //namespace