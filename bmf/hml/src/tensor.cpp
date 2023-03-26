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

#include <hmp/tensor.h>
#include <hmp/core/stream.h>
#include <kernel/tensor_factory.h>
#include <kernel/unary_ops.h>
#include <kernel/binary_ops.h>
#include <kernel/shape_transform.h>
#include <tensor_utils.h>
#include <hmp/format.h>

namespace hmp{


Tensor from_buffer(DataPtr &&data, ScalarType scalarType, 
    const SizeArray &shape, const optional<SizeArray> &strides)
{
    auto nitems = TensorInfo::calcNumel(shape);
    auto buffer = Buffer(std::move(data), scalarType, nitems, nullptr);
    checkSizeArray(shape, "from_buffer");
    if(strides){
        return makeRefPtr<TensorInfo>(buffer, shape, strides.value());
    }
    else{
        return makeRefPtr<TensorInfo>(buffer, shape);
    }
}

/////////

std::string Tensor::repr() const
{
    if(defined()){
        return fmt::format("Tensor({}, {}, {})", 
            device(),
            scalar_type(),
            shape());
    }
    else{
        return "Tensor(Undefined)";
    }

}

Tensor& Tensor::fill_(const Scalar &value)
{
    return hmp::fill(*this, value);
}

Tensor Tensor::to(const Device &device, bool non_blocking) const
{
    Tensor out;
    if(this->device() == device){
        out = alias();
    }
    else{
        auto options = this->options().device(device);
        if(device.type() == kCPU && non_blocking){
            options = options.pinned_memory(true);
        }
        auto tmp = empty_like(*this, options);
        out = copy(tmp, *this);
    }

    return out;
}


Tensor Tensor::to(DeviceType device, bool non_blocking) const
{
    return to(Device(device), non_blocking);
}

Tensor Tensor::to(ScalarType dtype) const
{
    if(this->dtype() == dtype){
        return alias();
    }
    else{
        auto tmp = empty_like(
            *this, 
            this->options().dtype(dtype));
        copy(tmp, *this);
        return tmp;
    }
}


Tensor& Tensor::copy_(const Tensor &src)
{
    copy(*this, src);
    return *this;
}


Tensor Tensor::as_strided(const SizeArray &shape, const SizeArray &strides, optional<int64_t> offset) const
{
    auto self = alias();
    self.as_strided_(shape, strides, offset);
    return self;
}

Tensor& Tensor::as_strided_(const SizeArray &shape, const SizeArray &strides, optional<int64_t> offset_)
{
    checkSizeArray(shape, "as_strides_");

    auto offset = offset_.value_or(tensorInfo()->bufferOffset());
    tensorInfo()->setSizesAndStrides(shape, strides, offset);
    return *this;
}


Tensor Tensor::expand(const SizeArray &shape) const
{
    auto result = inferExpandGeometry(this->shape(), this->strides(), shape);
    return as_strided(result.shape, result.strides);
}


Tensor Tensor::expand_as(const Tensor &other) const
{
    return expand(other.shape());
}


Tensor& Tensor::squeeze_(optional<int64_t> dim)
{
    if(dim){
        auto result = inferSqueezeGeometry(*this,
             wrap_size(dim.value(), this->dim()));
        return as_strided_(result.shape, result.strides);
    }
    else{
        auto result = inferSqueezeGeometry(*this);
        return as_strided_(result.shape, result.strides);
    }
}

Tensor Tensor::squeeze(optional<int64_t> dim) const
{
    auto out = alias();
    return out.squeeze_(dim);
}

Tensor& Tensor::unsqueeze_(int64_t dim_)
{
    auto dim = wrap_size(dim_, this->dim() + 1);
    auto result = inferUnsqueezeGeometry(*this, dim);
    return as_strided_(result.shape, result.strides);
}


Tensor Tensor::unsqueeze(int64_t dim) const
{
    auto out = alias();
    return out.unsqueeze_(dim);
}


Tensor Tensor::alias() const
{
    auto out = makeRefPtr<TensorInfo>(
        tensorInfo()->buffer(),
        shape(),
        strides(),
        tensorInfo()->bufferOffset());

    return Tensor(std::move(out));
}

Tensor Tensor::view(const SizeArray &shape_) const
{
    auto shape = inferSize(shape_, nitems());
    auto strides_ = computeStride(this->shape(), this->strides(), shape);
    HMP_REQUIRE(strides_, "can not view tensor as {} from {}", shape_, this->shape());

    return as_strided(shape, strides_.value());
}

Tensor Tensor::clone() const
{
    auto tmp = empty_like(*this, this->options());
    copy(tmp, *this);
    return tmp;
}


Tensor Tensor::reshape(const SizeArray &shape_) const
{
    auto shape = inferSize(shape_, nitems());
    auto strides = computeStride(this->shape(), this->strides(), shape);
    if(strides){
        return view(shape_);
    }
    else{
        auto tmp = clone();
        return tmp.view(shape);
    }
}


Tensor Tensor::transpose(int64_t dim0, int64_t dim1) const
{
    dim0 = wrap_size(dim0, dim());
    dim1 = wrap_size(dim1, dim());

    HMP_REQUIRE(dim0 < dim(), "transpose: dim0({}) is out of range {}", dim0, dim());
    HMP_REQUIRE(dim1 < dim(), "transpose: dim1({}) is out of range {}", dim1, dim());

    auto shape = this->shape();
    auto strides = this->strides();
    std::swap(shape[dim0], shape[dim1]);
    std::swap(strides[dim0], strides[dim1]);

    return as_strided(shape, strides);
}

Tensor Tensor::permute(const SizeArray &dims) const
{
    HMP_REQUIRE(dims.size() == this->dim(),
         "permute: invalid dim={}, expect {}", dims.size(), this->dim());

    SizeArray flags(this->dim(), 0);
    SizeArray newShape(this->dim(), 0), newStrides(this->dim(), 0);
    for(size_t i = 0; i < this->dim(); ++i){
        auto dim = wrap_size(dims[i], this->dim());
        HMP_REQUIRE(dim < this->dim(),
             "permute: invalid dim={} at {}, expect less than {}",
             dim, i, this->dim()
             );
        HMP_REQUIRE(!flags[dim], "permute: duplicate dim={} at {} found", dim, i);

        newShape[i] = this->size(dim);
        newStrides[i] = this->stride(dim);
        flags[dim] = 1;
    }

    return as_strided(newShape, newStrides);
}

Tensor Tensor::slice(int64_t dim, int64_t start, optional<int64_t> end_, int64_t step) const
{
    dim = wrap_size(dim, this->dim());
    HMP_REQUIRE(dim < this->dim(), "slice: dim({}) is out of range {}", dim, this->dim());
    HMP_REQUIRE(step > 0, "slice: required step > 0, got step={}", step);

    auto end = end_.value_or(size(dim));
    start = wrap_size(start, size(dim));
    end = wrap_size(end, size(dim));
    HMP_REQUIRE(start < size(dim),
         "slice: start {} is out of range, need less than {}", start, size(dim));
    HMP_REQUIRE(end <= size(dim), 
        "slice: start {} is out of range, need less or equal to {}", end, size(dim));
    auto size = (end + step - 1 - start) / step; //divup
    HMP_REQUIRE(size > 0, 
        "slice: expect at least 1 row selected, start={}, end={}, step={}", start, end, step);

    auto newBufferOffset = tensorInfo()->bufferOffset() + start * stride(dim);
    auto newShape = this->shape();
    newShape[dim] = size;
    auto newStrides = this->strides();
    newStrides[dim] *= step;

    return as_strided(newShape, newStrides, newBufferOffset);
}


Tensor Tensor::select(int64_t dim, int64_t index) const
{
    dim = wrap_size(dim, this->dim());
    HMP_REQUIRE(dim < this->dim(), "select : dim({}) is out of range {}", dim, this->dim());

    index = wrap_size(index, size(dim));
    HMP_REQUIRE(index < size(dim),
         "select: index {} is out of range, need less than {}", index, size(dim));

    auto out = slice(dim, index, index + 1);

    //squeeze
    auto newShape = out.shape();
    auto newStrides = out.strides();
    HMP_REQUIRE(newShape[dim] == 1, "select: internal error");
    newShape.erase(newShape.begin() + dim);
    newStrides.erase(newStrides.begin() + dim);

    out.tensorInfo()->setSizesAndStrides(
        newShape, newStrides, 
        out.tensorInfo()->bufferOffset());

    return out;
}


//factory functions
Tensor empty(const SizeArray &shape, const TensorOptions &options)
{
    checkSizeArray(shape, "empty");
    DeviceGuard dguard(options.device());

    return kernel::empty(shape, options);
}

Tensor empty_like(const Tensor &other, const optional<TensorOptions> &options_)
{
    auto options = options_.value_or(other.options());
    return kernel::empty(other.shape(), options);
}

Tensor zeros(const SizeArray &shape, const TensorOptions &options)
{
    return empty(shape, options).fill_(0);
}

Tensor zeros_like(const Tensor &other, const optional<TensorOptions> &options_)
{
    auto options = options_.value_or(other.options());
    return zeros(other.shape(), options);
}


Tensor ones(const SizeArray &shape, const TensorOptions &options)
{
    return empty(shape, options).fill_(1);
}

Tensor ones_like(const Tensor &other, const optional<TensorOptions> &options_)
{
    auto options = options_.value_or(other.options());
    return ones(other.shape(), options);
}

Tensor& fill(Tensor &self, const Scalar& value)
{
    using namespace kernel;
#ifndef HMP_BUILD_SHARED
    HMP_IMPORT_DEVICE_DISPATCH(kCPU, fill_stub);
#ifdef HMP_EANBLE_CUDA
    HMP_IMPORT_DEVICE_DISPATCH(kCUDA, fill_stub);
#endif
#endif

    kernel::fill_stub(self.device_type(), self, value);

    return self;
}


Tensor arange(int64_t start, int64_t end, int64_t step, const TensorOptions &options)
{
    using namespace kernel;
#ifndef HMP_BUILD_SHARED
    HMP_IMPORT_DEVICE_DISPATCH(kCPU, fill_stub);
#ifdef HMP_EANBLE_CUDA
    HMP_IMPORT_DEVICE_DISPATCH(kCUDA, fill_stub);
#endif
#endif

    HMP_REQUIRE(start < end, "arange: expect start < end, got start={}, end={}", start, end);
    HMP_REQUIRE(step > 0, "arange: expect step > 0, got step={}", step);

    auto size = (end - start + step - 1) / step; //divup
    auto out = hmp::empty({size}, options);
    kernel::arange_stub(out.device_type(), out, start, end, step);
    return out;
}

Tensor Tensor::contiguous() const
{
    if(this->is_contiguous()){
        return *this;
    }
    else{
        return clone();
    }
}


////// Unary ops

#define DEFINE_TENSOR_UOPS(op) \
    Tensor Tensor::op() const \
    { \
        auto out = empty_like(*this, this->options()); \
        kernel::op(out, *this); \
        return out; \
    } \
    Tensor& Tensor::op##_()\
    { \
        kernel::op(*this, *this); \
        return *this; \
    }


DEFINE_TENSOR_UOPS(round)
DEFINE_TENSOR_UOPS(ceil)
DEFINE_TENSOR_UOPS(floor)
DEFINE_TENSOR_UOPS(abs)


Tensor Tensor::clip(const Scalar &min, const Scalar &max) const
{
    auto out = empty_like(*this, this->options());
    kernel::clip(out, *this, min, max);
    return out;
}

Tensor& Tensor::clip_(const Scalar &min, const Scalar &max)
{
    kernel::clip(*this, *this, min, max);
    return *this;
}


////// Binary ops

#define DEFINE_TENSOR_BOPS(name, op) \
    Tensor Tensor::name(const Tensor &b) const{    \
        auto out = empty_like(*this, this->options()); \
        kernel::name(out, *this, b); \
        return out; \
    } \
    Tensor& Tensor::name##_(const Tensor &b){    \
        kernel::name(*this, *this, b); \
        return *this; \
    } \
    Tensor Tensor::operator op(const Tensor &b) const{    \
        return name(b); \
    } \
    Tensor& Tensor::operator op##=(const Tensor &b){    \
        return name##_(b); \
    } \
    Tensor Tensor::name(const Scalar &b) const{    \
        auto out = empty_like(*this, this->options()); \
        kernel::name(out, *this, b); \
        return out; \
    } \
    Tensor& Tensor::name##_(const Scalar &b){    \
        kernel::name(*this, *this, b); \
        return *this; \
    } \
    Tensor Tensor::operator op(const Scalar &b) const{    \
        return name(b); \
    } \
    Tensor& Tensor::operator op##=(const Scalar &b){    \
        return name##_(b); \
    } 


DEFINE_TENSOR_BOPS(mul, *)
DEFINE_TENSOR_BOPS(add, +)
DEFINE_TENSOR_BOPS(sub, -)
DEFINE_TENSOR_BOPS(div, /)


Tensor operator*(const Scalar &a, const Tensor &b)
{
    return b * a;
}

Tensor operator+(const Scalar &a, const Tensor &b)
{
    return b + a;
}

Tensor operator-(const Scalar &a, const Tensor &b)
{
    auto out = empty_like(b, b.options());
    kernel::sub(out, a, b);
    return out;
}

Tensor operator/(const Scalar &a, const Tensor &b)
{
    auto out = empty_like(b, b.options());
    kernel::div(out, a, b);
    return out;
}

// shape transform
Tensor Tensor::flatten() const
{
    return reshape({-1});
}


void Tensor::tofile(const std::string &fn)
{
    hmp::tofile(*this, fn);
}

//copy functions
Tensor& copy(Tensor &self, const Tensor &other)
{
    return kernel::copy(self, other);
}


//shape transformation
Tensor concat(const TensorList &tensors, int64_t axis)
{
    return kernel::concat(tensors, axis);
}

Tensor& concat(Tensor &out, const TensorList &tensors, int64_t axis)
{
    return kernel::concat(out, tensors, axis);
}

Tensor stack(const TensorList &tensors, int64_t axis)
{
    return kernel::stack(tensors, axis);
}

Tensor& stack(Tensor &out, const TensorList &tensors, int64_t axis)
{
    return kernel::stack(out, tensors, axis);
}

Tensor vstack(const TensorList &tensors)
{
    return kernel::vstack(tensors);
}

Tensor& vstack(Tensor &out, const TensorList &tensors)
{
    return kernel::vstack(out, tensors);
}

Tensor hstack(const TensorList &tensors)
{
    return kernel::hstack(tensors);
}

Tensor& hstack(Tensor &out, const TensorList &tensors)
{
    return kernel::hstack(out, tensors);
}


#if defined(__ANDROID__) && HMP_NDK_VERSION < 24
#define FSEEK fseek
#define FTELL ftell
#else
#define FSEEK fseeko
#define FTELL ftello
#endif

Tensor fromfile(const std::string &fn, ScalarType dtype, int64_t count, int64_t offset)
{
    auto fp = std::shared_ptr<FILE>(fopen(fn.c_str(), "rb"), fclose);
    HMP_REQUIRE(fp, "Open file {} failed", fn);

    FSEEK(fp.get(), 0, SEEK_END);
    auto size = FTELL(fp.get());

    auto itemsize = sizeof_scalar_type(dtype);
    offset *= itemsize;
    //
    if(FSEEK(fp.get(), offset, SEEK_SET) < 0){
        throw std::runtime_error("Invalid file offset");
    }

    //
    int64_t nitems = (size - offset) / itemsize;
    nitems = count > 0 ? std::min<int64_t>(nitems, count) : nitems;

    auto data = empty(SizeArray{nitems}, dtype);
    auto nread = fread(data.unsafe_data(), itemsize, nitems, fp.get());
    HMP_REQUIRE(nread == nitems, "Internal error");

    return data;
}

void tofile(const Tensor &data, const std::string &fn)
{
    auto fp = std::shared_ptr<FILE>(fopen(fn.c_str(), "wb"), fclose);
    HMP_REQUIRE(fp, "Open file {} failed", fn);

    auto tmp = data.cpu().contiguous();
    auto nwrite = fwrite(tmp.unsafe_data(), tmp.itemsize(), tmp.nitems(), fp.get());
    HMP_REQUIRE(nwrite == tmp.nitems(), "write data to file failed, errno={} {}, {}",
                 errno, nwrite, tmp.nitems());
}

} //namespace