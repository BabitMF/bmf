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

#include <hmp/core/tensor_info.h>

// #include <pybind11/pybind11.h>

// namespace py = pybind11;
namespace hmp {

class HMP_API Tensor {
    RefPtr<TensorInfo> self_;

  public:
    Tensor() {}

    Tensor(RefPtr<TensorInfo> &&info) : self_(std::move(info)) {}

    //
    const RefPtr<TensorInfo> &tensorInfo() const { return self_; }

    // attributes
    std::string repr() const;

    Tensor clone() const;

    Tensor alias() const;

    // view operations
    Tensor view(const SizeArray &shape) const;
    Tensor as_strided(const SizeArray &shape, const SizeArray &strides,
                      optional<int64_t> offset = nullopt) const;
    Tensor &as_strided_(const SizeArray &shape, const SizeArray &strides,
                        optional<int64_t> offset = nullopt);

    Tensor expand(const SizeArray &shape) const;
    Tensor expand_as(const Tensor &other) const;

    Tensor squeeze(optional<int64_t> dim = nullopt) const;
    Tensor &squeeze_(optional<int64_t> dim = nullopt);
    Tensor unsqueeze(int64_t dim = 0) const;
    Tensor &unsqueeze_(int64_t dim = 0);

    Tensor transpose(int64_t dim0, int64_t dim1) const;
    Tensor permute(const SizeArray &dims) const;
    Tensor slice(int64_t dim, int64_t start, optional<int64_t> end = nullopt,
                 int64_t step = 1) const;
    Tensor select(int64_t dim, int64_t index) const;

    //
    Tensor reshape(const SizeArray &shape) const;

    //
    inline TensorOptions options() const { return self_->options(); }
    inline bool defined() const { return self_; }
    inline explicit operator bool() const { return defined(); }

    inline const Device &device() const { return self_->device(); }

    inline DeviceType device_type() const { return self_->device_type(); }
    inline int64_t device_index() const { return self_->device_index(); }

    inline ScalarType scalar_type() const { return self_->scalar_type(); }
    inline ScalarType dtype() const { return scalar_type(); }

    inline const SizeArray &shape() const { return self_->shape(); }

    inline const SizeArray &strides() const { return self_->strides(); }

    inline int64_t dim() const { return self_->dim(); }

    inline int64_t size(int64_t dim) const {
        dim = wrap_size(dim, this->dim());
        HMP_REQUIRE(dim < this->dim(), "dim {} is out of range {}", dim,
                    this->dim());
        return shape()[dim];
    }

    inline int64_t stride(int64_t dim) const {
        dim = wrap_size(dim, this->dim());
        HMP_REQUIRE(dim < this->dim(), "dim {} is out of range {}", dim,
                    this->dim());
        return strides()[dim];
    }

    inline int64_t nbytes() const { return self_->nbytes(); }

    inline int64_t itemsize() const { return self_->itemsize(); }

    inline int64_t nitems() const { return self_->nitems(); }

    inline bool is_contiguous() const { return self_->is_contiguous(); };

    inline bool is_same(const Tensor &other) {
        return self_.get() == other.self_.get();
    }

    inline bool is_cpu() const { return self_->device_type() == kCPU; }
    inline bool is_cuda() const { return self_->device_type() == kCUDA; }

    inline Tensor cpu(bool non_blocking = false) const {
        return to(kCPU, non_blocking);
    }
    inline Tensor cuda() const { return to(kCUDA); }

    template <typename T> inline T *data() const {
        HMP_REQUIRE(defined(), "Tensor is not defined");
        return self_->data<T>();
    }

    inline void *unsafe_data() const { return self_->unsafe_data(); }

    //
    Tensor &fill_(const Scalar &value);

    //
    Tensor to(DeviceType device, bool non_blocking = false) const;
    Tensor to(const Device &device, bool non_blocking = false) const;
    Tensor to(ScalarType dtype) const;

    Tensor &copy_(const Tensor &src);

    //
    Tensor contiguous() const;

    // unary ops
    Tensor round() const;
    Tensor &round_();
    Tensor ceil() const;
    Tensor &ceil_();
    Tensor floor() const;
    Tensor &floor_();
    Tensor abs() const;
    Tensor &abs_();
    Tensor clip(const Scalar &min, const Scalar &max) const;
    Tensor &clip_(const Scalar &min, const Scalar &max);

// binary ops
#define DECLARE_TENSOR_BOP(name, op)                                           \
    Tensor name(const Tensor &b) const;                                        \
    Tensor &name##_(const Tensor &b);                                          \
    Tensor operator op(const Tensor &b) const;                                 \
    Tensor &operator op##=(const Tensor &b);                                   \
    Tensor name(const Scalar &b) const;                                        \
    Tensor &name##_(const Scalar &b);                                          \
    Tensor operator op(const Scalar &b) const;                                 \
    Tensor &operator op##=(const Scalar &b);                                   \
    HMP_API friend Tensor operator op(const Scalar &a, const Tensor &b);

    DECLARE_TENSOR_BOP(mul, *)
    DECLARE_TENSOR_BOP(add, +)
    DECLARE_TENSOR_BOP(sub, -)
    DECLARE_TENSOR_BOP(div, /)

#undef DECLARE_TENSOR_BOP

    // shape transform
    Tensor flatten() const;

    // TODO
    // void used_in_stream(const Stream &stream);

    // File IO
    void tofile(const std::string &fn);
};

using TensorList = std::vector<Tensor>;

//////////////////
HMP_API std::string stringfy(const Tensor &tensor);

// factory functions
HMP_API Tensor from_buffer(DataPtr &&data, ScalarType scalar_type,
                           const SizeArray &shape,
                           const optional<SizeArray> &strides = nullopt);
HMP_API Tensor empty(const SizeArray &shape, const TensorOptions &options = {});
HMP_API Tensor empty_like(const Tensor &other,
                          const optional<TensorOptions> &options = nullopt);
HMP_API Tensor zeros(const SizeArray &shape, const TensorOptions &options = {});
HMP_API Tensor zeros_like(const Tensor &other,
                          const optional<TensorOptions> &options = nullopt);
HMP_API Tensor ones(const SizeArray &shape, const TensorOptions &options = {});
HMP_API Tensor ones_like(const Tensor &other,
                         const optional<TensorOptions> &options = nullopt);
HMP_API Tensor arange(int64_t start, int64_t end, int64_t step = 1,
                      const TensorOptions &options = {});
HMP_API Tensor &fill(Tensor &output, const Scalar &value);

// copy functions
HMP_API Tensor &copy(Tensor &dst, const Tensor &src);

// shape transformation
HMP_API Tensor concat(const TensorList &tensors, int64_t axis = 0);
HMP_API Tensor &concat(Tensor &out, const TensorList &tensors,
                       int64_t axis = 0);
HMP_API Tensor stack(const TensorList &tensors, int64_t axis = 0);
HMP_API Tensor &stack(Tensor &out, const TensorList &tensors, int64_t axis = 0);
HMP_API Tensor vstack(const TensorList &tensors);
HMP_API Tensor &vstack(Tensor &out, const TensorList &tensors);
HMP_API Tensor hstack(const TensorList &tensors);
HMP_API Tensor &hstack(Tensor &out, const TensorList &tensors);

//
HMP_API Tensor fromfile(const std::string &fn, ScalarType dtype = kFloat32,
                        int64_t count = -1, int64_t offset = 0);
HMP_API void tofile(const Tensor &data, const std::string &fn);

} // namespace hmp
