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

#include <hmp/core/logging.h>
#include <hmp/core/macros.h>
#include <atomic>

namespace hmp {

class HMP_API RefObject {
    // TODO: add weak ref support
    mutable std::atomic<int> refcount_;

    template <typename T> friend class RefPtr;

  protected:
    constexpr RefObject() : refcount_(0) {}
    RefObject(RefObject &&other) : RefObject() {}
    RefObject(const RefObject &) : RefObject() {}
    virtual ~RefObject();

  private:
    // this function is called when refcount == 0
    // for future weak ref support
    virtual void destroy() {}
};

template <typename T> class RefPtr final {
    T *self_ = nullptr;

    explicit RefPtr(T *self) noexcept : self_(self) {}

  public:
    template <typename U> friend class RefPtr;

    using element_type = T;

    RefPtr() noexcept : RefPtr(nullptr) {}
    RefPtr(RefPtr &&rhs) noexcept : self_(rhs.self_) { rhs.self_ = nullptr; }

    template <typename U> RefPtr(RefPtr<U> &&rhs) noexcept : self_(rhs.self_) {
        static_assert(std::is_convertible<U *, T *>::value,
                      "Invalid type conversion");
        rhs.self_ = nullptr;
    }

    RefPtr(const RefPtr &rhs) : self_(rhs.self_) { inc_ref(self_); }

    template <typename U> RefPtr(const RefPtr<U> &rhs) : self_(rhs.self_) {
        static_assert(std::is_convertible<U *, T *>::value,
                      "Invalid type conversion");
        inc_ref(self_);
    }

    inline RefPtr &operator=(RefPtr &&rhs) {
        auto tmp = RefPtr(std::move(rhs));
        swap(tmp);
        return *this;
    }

    template <typename U> inline RefPtr &operator=(RefPtr<U> &&rhs) {
        auto tmp = RefPtr<T>(std::move(rhs));
        swap(tmp);
        return *this;
    }

    inline RefPtr &operator=(const RefPtr &rhs) {
        auto tmp = RefPtr(rhs);
        swap(tmp);
        return *this;
    }

    template <typename U> inline RefPtr &operator=(const RefPtr<U> &rhs) {
        auto tmp = RefPtr<T>(rhs);
        swap(tmp);
        return *this;
    }

    T &operator*() { return *self_; }

    const T &operator*() const { return *self_; }

    ~RefPtr() { reset(); }

    inline T *get() const noexcept { return self_; }

    inline T *operator->() const noexcept { return self_; }

    inline T &operator&() const noexcept { return *self_; }

    inline operator bool() const noexcept { return get() != nullptr; }

    void reset() {
        dec_ref(self_);

        self_ = nullptr;
    }

    void swap(RefPtr &rhs) { std::swap(self_, rhs.self_); }

    inline int refcount() const noexcept {
        if (self_) {
            return self_->refcount_.load();
        } else {
            return 0;
        }
    }

    inline bool unique() const noexcept { return refcount() == 1; }

    inline T *release() noexcept {
        T *result = nullptr;
        std::swap(result, self_);
        return result;
    }

    template <typename U> RefPtr<U> cast() {
        auto p = dynamic_cast<U *>(self_);
        auto ret = RefPtr<U>(p);
        if (p) {
            ++p->refcount_;
        }
        return ret;
    }

    // take it and increase the refcount if inc==true
    static RefPtr take(T *self, bool inc = false) {
        if (inc)
            ++self->refcount_;
        return RefPtr(self);
    }

    static T *inc_ref(T *obj) {
        if (obj) {
            auto refcount = ++obj->refcount_;
            HMP_REQUIRE(
                refcount != 1,
                "RefPtr: can't increase refcount after it reach zeros.");
        }

        return obj;
    }

    static void dec_ref(T *obj) {
        if (obj != nullptr && --obj->refcount_ == 0) {
            obj->destroy();

            delete obj;
        }
    }

    template <typename... Args> static RefPtr make(Args &&...args) {
        auto result = RefPtr(new T(std::forward<Args>(args)...));
        ++result.self_->refcount_;
        return result;
    }
};

template <typename T, typename... Args>
inline RefPtr<T> makeRefPtr(Args &&...args) {
    return RefPtr<T>::make(std::forward<Args>(args)...);
}

template <typename T> T *inc_ref(T *obj) { return RefPtr<T>::inc_ref(obj); }

template <typename T> void dec_ref(T *obj) { RefPtr<T>::dec_ref(obj); }

} // namespace hmp