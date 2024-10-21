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

#include <vector>
#include <hmp/core/ref_ptr.h>
#include <hmp/core/buffer.h>
#include <hmp/core/scalar.h>

namespace hmp {

using SizeArray = std::vector<int64_t>;

inline std::string stringfy(const SizeArray &array) {
    return fmt::format("({})", fmt::join(array, ", "));
}

inline SizeArray calcContiguousStrides(const SizeArray &shape) {
    SizeArray strides(shape.size());
    if (shape.size()) {
        strides[shape.size() - 1] = 1;
        for (size_t i = 1; i < shape.size(); ++i) {
            auto j = shape.size() - i - 1;
            strides[j] = strides[j + 1] * shape[j + 1];
        }
    }

    return strides;
}

// we won't check if idx is out of range
inline int64_t wrap_size(int64_t idx, int64_t size) {
    return idx >= 0 ? idx : size + idx;
}

class HMP_API TensorOptions {
  public:
    TensorOptions() = default;
    TensorOptions(const TensorOptions &) = default;

    TensorOptions(ScalarType stype) : scalar_type_(stype) {}

    TensorOptions(const Device &device) : device_(device) {}

    TensorOptions(DeviceType device) : device_(device) {}

    TensorOptions &scalar_type(ScalarType stype) {
        scalar_type_ = stype;
        return *this;
    }

    TensorOptions &dtype(ScalarType stype) {
        scalar_type_ = stype;
        return *this;
    }

    TensorOptions &device(const Device &device) {
        device_ = device;
        return *this;
    }

    TensorOptions &pinned_memory(bool pinned) {
        pinned_memory_ = pinned;
        return *this;
    }

    const Device &device() const { return device_; }

    ScalarType scalar_type() const { return scalar_type_; }

    ScalarType dtype() const { return scalar_type_; }

    bool pinned_memory() const { return pinned_memory_; }

    // generic programing interfaces
    void option(const ScalarType &dtype) { scalar_type(dtype); }

    void option(const Device &device_) { device(device_); }

    void option(const std::string &device_) { device(device_); }

    void option(const char *device_) { device(Device(device_)); }

    void option(bool pinned) { pinned_memory(pinned); }

    TensorOptions &options() { return *this; }

    template <typename T, typename... Args>
    TensorOptions &options(T v, Args &&...args) {
        option(v);
        return options(std::forward<Args>(args)...);
    }

    template <typename... Args> static TensorOptions make(Args &&...args) {
        return TensorOptions().options(std::forward<Args>(args)...);
    }

  private:
    bool pinned_memory_ = false;
    ScalarType scalar_type_ = kFloat32;
    Device device_ = kCPU;
};

class HMP_API TensorInfo final : public RefObject {
  public:
    TensorInfo() = delete;
    TensorInfo(const TensorInfo &) = default;
    TensorInfo(TensorInfo &&) = default;

    TensorInfo(const Buffer &buffer, const SizeArray &shape,
               int64_t bufferOffset = 0);
    TensorInfo(const Buffer &buffer, const SizeArray &shape,
               const SizeArray &strides, int64_t bufferOffset = 0);

    void setSizesAndStrides(const SizeArray &shape, int64_t bufferOffset = 0);
    void setSizesAndStrides(const SizeArray &shape, const SizeArray &strides,
                            int64_t bufferOffset = 0);

    static inline int64_t calcNumel(const SizeArray &shape) {
        int64_t n = 1; // scalar tensor with nitems==1
        for (auto &s : shape) {
            n *= s;
        }
        return n;
    }

    inline TensorOptions options() const {
        return TensorOptions(device())
            .scalar_type(scalar_type())
            .pinned_memory(pinned_memory());
    }

    inline const Device &device() const { return buffer_.device(); }

    inline DeviceType device_type() const { return device().type(); }
    inline int64_t device_index() const { return device().index(); }

    inline ScalarType scalar_type() const { return buffer_.scalar_type(); }

    inline bool pinned_memory() const { return buffer_.pinned_memory(); }

    inline const SizeArray &shape() const { return shape_; }

    inline const SizeArray &strides() const { return strides_; }

    inline const Buffer &buffer() const { return buffer_; }

    inline int64_t dim() const { return shape_.size(); }

    inline int64_t nbytes() const { return nitems_ * itemsize(); }

    inline int64_t itemsize() const { return buffer_.itemsize(); }

    inline int64_t nitems() const { return nitems_; }

    inline int64_t bufferOffset() const { return bufferOffset_; }

    bool is_contiguous() const;

    template <typename T> inline T *data() const {
        HMP_REQUIRE(getScalarType<T>() == scalar_type(),
                    "Invalid scalar type {}, expect {}", getScalarType<T>(),
                    scalar_type());
        return static_cast<T *>(unsafe_data());
    }

    inline void *unsafe_data() const {
        return static_cast<char *>(buffer_.unsafe_data()) +
               bufferOffset_ * itemsize();
    }

  private:
    Buffer buffer_;
    int64_t bufferOffset_;
    SizeArray shape_;
    SizeArray strides_;
    int64_t nitems_;
};

} // namespace hmp

// for SizeAarray
template <> struct fmt::formatter<hmp::SizeArray> {
    template <typename ParseContext> constexpr auto parse(ParseContext &ctx) {
        return ctx.begin();
    }

    auto format(hmp::SizeArray c, fmt::format_context &ctx) {
        return fmt::format_to(ctx.out(), "({})", fmt::join(c, ", "));
    }
};
