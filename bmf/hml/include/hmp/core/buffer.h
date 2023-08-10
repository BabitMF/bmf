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

#include <hmp/core/ref_ptr.h>
#include <hmp/core/allocator.h>
#include <hmp/core/scalar_type.h>

namespace hmp {

class HMP_API BufferImpl final : public RefObject {
  public:
    BufferImpl() = delete;
    BufferImpl(BufferImpl &&) = default;
    BufferImpl(const BufferImpl &) = delete;

    BufferImpl(DataPtr &&ptr, ScalarType scalar_type, int64_t nitems,
               Allocator *allocator, bool pinned_memory = false)
        : pinned_memory_(pinned_memory), data_(std::move(ptr)),
          allocator_(allocator), numel_(nitems), scalarType_(scalar_type) {
        HMP_REQUIRE(data_, "Empty data is not supported");
    }

    BufferImpl(ScalarType scalar_type, int64_t nitems, Allocator *allocator,
               bool pinned_memory = false)
        : pinned_memory_(pinned_memory), allocator_(allocator), numel_(nitems),
          scalarType_(scalar_type) {
        HMP_REQUIRE(allocator != nullptr,
                    "Buffer can not be initialize without allocator and data");
        data_ = allocator->alloc(nitems * sizeof_scalar_type(scalar_type));
    }

    inline bool pinned_memory() const { return pinned_memory_; }

    inline ScalarType scalar_type() const { return scalarType_; }

    inline int64_t nitems() const { return numel_; }

    inline int64_t itemsize() const { return sizeof_scalar_type(scalarType_); }

    inline int64_t nbytes() const { return nitems() * itemsize(); }

    inline const Device &device() const { return data_.device(); }

    inline Allocator *allocator() const { return allocator_; }

    template <typename T> T *data() const {
        auto scalar_type = getScalarType<T>();
        HMP_REQUIRE(
            scalar_type == scalarType_,
            "Try to access buffer with data type {}, but it have data type {}",
            scalar_type, scalarType_);
        return static_cast<T *>(data_.get());
    }

    void *unsafe_data() const { return data_.get(); }

  private:
    bool pinned_memory_;
    DataPtr data_;
    Allocator *allocator_;
    int64_t numel_;
    ScalarType scalarType_;
};

class HMP_API Buffer {
    RefPtr<BufferImpl> self_;

  public:
    Buffer() = default;
    Buffer(const Buffer &) = default;
    Buffer(Buffer &&) = default;
    Buffer &operator=(const Buffer &other) {
        self_ = other.self_;
        return *this;
    }

    Buffer &operator=(Buffer &&other) {
        self_ = std::move(other.self_);
        return *this;
    }

    Buffer(DataPtr &&ptr, ScalarType scalar_type, int64_t nitems,
           Allocator *allocator, bool pinned_memory = false)
        : self_(makeRefPtr<BufferImpl>(std::move(ptr), scalar_type, nitems,
                                       allocator, pinned_memory)) {}

    Buffer(ScalarType scalar_type, int64_t nitems, Allocator *allocator,
           bool pinned_memory = false)
        : self_(makeRefPtr<BufferImpl>(scalar_type, nitems, allocator,
                                       pinned_memory)) {}

    bool defined() const { return self_; }

    inline bool pinned_memory() const { return self_->pinned_memory(); }

    inline ScalarType scalar_type() const { return self_->scalar_type(); }

    inline int64_t nitems() const { return self_->nitems(); }

    inline int64_t itemsize() const { return self_->itemsize(); }

    inline int64_t nbytes() const { return self_->nbytes(); }

    inline Allocator *allocator() const { return self_->allocator(); }

    inline const Device &device() const { return self_->device(); }

    inline int refcount() const { return self_.refcount(); }

    template <typename T> inline T *data() const { return self_->data<T>(); }

    inline void *unsafe_data() const { return self_->unsafe_data(); }
};

} // namespace hmp
