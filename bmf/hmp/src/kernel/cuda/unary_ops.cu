#include <kernel/unary_ops.h>
#include <kernel/cuda/kernel_utils.h>

namespace hmp{
namespace kernel{
namespace{

Tensor& round_cuda(Tensor &out, const Tensor &in)
{
    HMP_DISPATCH_FLOATING_POINT_TYPES_AND_HALF(in.scalar_type(), "round_cuda", [&](){
        cuda::uop_kernel<scalar_t, scalar_t>(out, in, [=]HMP_HOST_DEVICE(scalar_t v){
            return ::round(v);
        });
    });
    return out;
}


Tensor& ceil_cuda(Tensor &out, const Tensor &in)
{
    HMP_DISPATCH_FLOATING_POINT_TYPES_AND_HALF(in.scalar_type(), "ceil_cuda", [&](){
        cuda::uop_kernel<scalar_t, scalar_t>(out, in, [=]HMP_HOST_DEVICE(scalar_t v){
            return ::ceil(v);
        });
    });
    return out;
}


Tensor& floor_cuda(Tensor &out, const Tensor &in)
{
    HMP_DISPATCH_FLOATING_POINT_TYPES_AND_HALF(in.scalar_type(), "floor_cuda", [&](){
        cuda::uop_kernel<scalar_t, scalar_t>(out, in, [=]HMP_HOST_DEVICE(scalar_t v){
            return ::floor(v);
        });
    });
    return out;
}

Tensor& abs_cuda(Tensor &out, const Tensor &in)
{
    HMP_DISPATCH_ALL_TYPES_AND_HALF(in.scalar_type(), "abs_cuda", [&](){
        cuda::uop_kernel<scalar_t, scalar_t>(out, in, [=]HMP_HOST_DEVICE(scalar_t v){
            return ::abs(v);
        });
    });
    return out;
}


Tensor& minus_cuda(Tensor &out, const Tensor &in)
{
    HMP_DISPATCH_ALL_TYPES_AND_HALF(in.scalar_type(), "minus_cuda", [&](){
        cuda::uop_kernel<scalar_t, scalar_t>(out, in, [=]HMP_HOST_DEVICE(scalar_t v){
            return -v;
        });
    });
    return out;
}


Tensor& clip_cuda(Tensor &out, const Tensor &in, const Scalar &min, const Scalar &max)
{
    HMP_DISPATCH_ALL_TYPES_AND_HALF(in.scalar_type(), "clip_cuda", [&](){
        auto min_v = min.to<scalar_t>();
        auto max_v = max.to<scalar_t>();

        HMP_REQUIRE(min_v <= max_v,
             "clip_cuda: expect min <= max, got min={}, max={}", min_v, max_v);

        cuda::uop_kernel<scalar_t, scalar_t>(out, in, [=]HMP_HOST_DEVICE(scalar_t v){
            return v < min_v ? min_v : (v > max_v ? max_v : v);
        });
    });
    return out;
}



HMP_DEVICE_DISPATCH(kCUDA, round_stub, &round_cuda)
HMP_DEVICE_DISPATCH(kCUDA, ceil_stub, &ceil_cuda)
HMP_DEVICE_DISPATCH(kCUDA, floor_stub, &floor_cuda)
HMP_DEVICE_DISPATCH(kCUDA, abs_stub, &abs_cuda)
HMP_DEVICE_DISPATCH(kCUDA, minus_stub, &minus_cuda)
HMP_DEVICE_DISPATCH(kCUDA, clip_stub, &clip_cuda)


}}} //namespace