#include <kernel/binary_ops.h>
#include <kernel/cuda/kernel_utils.h>

namespace hmp{
namespace kernel{
namespace{

Tensor& mul_cuda(Tensor &out, const Tensor &ina, const Tensor &inb)
{
    HMP_DISPATCH_ALL_TYPES_AND_HALF(out.scalar_type(), "mul_cuda", [&](){
        cuda::bop_kernel<scalar_t, scalar_t, scalar_t>(out, ina, inb, 
            [=]HMP_HOST_DEVICE(scalar_t a, scalar_t b){
                return a * b;
        });
    });
    return out;
}


Tensor& mul_scalar_cuda(Tensor &out, const Tensor &ina, const Scalar &inb)
{
    HMP_DISPATCH_ALL_TYPES_AND_HALF(out.scalar_type(), "mul_scalar_cuda", [&](){
        auto b = inb.to<scalar_t>();
        cuda::uop_kernel<scalar_t, scalar_t>(out, ina,
            [=]HMP_HOST_DEVICE(scalar_t a){
                return a * b;
        });
    });
    return out;
}


Tensor& add_cuda(Tensor &out, const Tensor &ina, const Tensor &inb)
{
    HMP_DISPATCH_ALL_TYPES_AND_HALF(out.scalar_type(), "add_cuda", [&](){
        cuda::bop_kernel<scalar_t, scalar_t, scalar_t>(out, ina, inb, 
            [=]HMP_HOST_DEVICE(scalar_t a, scalar_t b){
                return a + b;
        });
    });
    return out;
}


Tensor& add_scalar_cuda(Tensor &out, const Tensor &ina, const Scalar &inb)
{
    HMP_DISPATCH_ALL_TYPES_AND_HALF(out.scalar_type(), "add_scalar_cuda", [&](){
        auto b = inb.to<scalar_t>();
        cuda::uop_kernel<scalar_t, scalar_t>(out, ina,
            [=]HMP_HOST_DEVICE(scalar_t a){
                return a + b;
        });
    });
    return out;
}


Tensor& sub_cuda(Tensor &out, const Tensor &ina, const Tensor &inb)
{
    HMP_DISPATCH_ALL_TYPES_AND_HALF(out.scalar_type(), "sub_cuda", [&](){
        cuda::bop_kernel<scalar_t, scalar_t, scalar_t>(out, ina, inb, 
            [=]HMP_HOST_DEVICE(scalar_t a, scalar_t b){
                return a - b;
        });
    });
    return out;
}


Tensor& sub_scalar_cuda(Tensor &out, const Tensor &ina, const Scalar &inb)
{
    HMP_DISPATCH_ALL_TYPES_AND_HALF(out.scalar_type(), "sub_scalar_cuda", [&](){
        auto b = inb.to<scalar_t>();
        cuda::uop_kernel<scalar_t, scalar_t>(out, ina,
            [=]HMP_HOST_DEVICE(scalar_t a){
                return a - b;
        });
    });
    return out;
}


Tensor& sub_scalar2_cuda(Tensor &out, const Scalar &ina, const Tensor &inb)
{
    HMP_DISPATCH_ALL_TYPES_AND_HALF(out.scalar_type(), "sub_scalar2_cuda", [&](){
        auto a = ina.to<scalar_t>();
        cuda::uop_kernel<scalar_t, scalar_t>(out, inb,
            [=]HMP_HOST_DEVICE(scalar_t b){
                return a - b;
        });
    });
    return out;
}


Tensor& div_cuda(Tensor &out, const Tensor &ina, const Tensor &inb)
{
    HMP_DISPATCH_ALL_TYPES_AND_HALF(out.scalar_type(), "div_cuda", [&](){
        cuda::bop_kernel<scalar_t, scalar_t, scalar_t>(out, ina, inb, 
            [=]HMP_HOST_DEVICE(scalar_t a, scalar_t b){
                return a / b;
        });
    });
    return out;
}


Tensor& div_scalar_cuda(Tensor &out, const Tensor &ina, const Scalar &inb)
{
    HMP_DISPATCH_ALL_TYPES_AND_HALF(out.scalar_type(), "div_scalar_cuda", [&](){
        auto b = inb.to<scalar_t>();
        cuda::uop_kernel<scalar_t, scalar_t>(out, ina,
            [=]HMP_HOST_DEVICE(scalar_t a){
                return a / b;
        });
    });
    return out;
}


Tensor& div_scalar2_cuda(Tensor &out, const Scalar &ina, const Tensor &inb)
{
    HMP_DISPATCH_ALL_TYPES_AND_HALF(out.scalar_type(), "div_scalar2_cuda", [&](){
        auto a = ina.to<scalar_t>();
        cuda::uop_kernel<scalar_t, scalar_t>(out, inb,
            [=]HMP_HOST_DEVICE(scalar_t b){
                return a / b;
        });
    });
    return out;
}



HMP_DEVICE_DISPATCH(kCUDA, mul_stub, &mul_cuda)
HMP_DEVICE_DISPATCH(kCUDA, mul_scalar_stub, &mul_scalar_cuda)
HMP_DEVICE_DISPATCH(kCUDA, add_stub, &add_cuda)
HMP_DEVICE_DISPATCH(kCUDA, add_scalar_stub, &add_scalar_cuda)
HMP_DEVICE_DISPATCH(kCUDA, sub_stub, &sub_cuda)
HMP_DEVICE_DISPATCH(kCUDA, sub_scalar_stub, &sub_scalar_cuda)
HMP_DEVICE_DISPATCH(kCUDA, sub_scalar_stub2, &sub_scalar2_cuda)
HMP_DEVICE_DISPATCH(kCUDA, div_stub, &div_cuda)
HMP_DEVICE_DISPATCH(kCUDA, div_scalar_stub, &div_scalar_cuda)
HMP_DEVICE_DISPATCH(kCUDA, div_scalar_stub2, &div_scalar2_cuda)

}}} //namespace