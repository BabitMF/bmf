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

#include <hmp/core/macros.h>
#include <hmp/core/scalar_type.h>

namespace hmp {

class HMP_API Scalar {
  public:
    Scalar() : Scalar(int64_t(0)) {}

#define DEFINE_INTEGRAL_CTOR(T, _)                                             \
    Scalar(T v) {                                                              \
        type_ = T_Int;                                                         \
        data_.i = v;                                                           \
    }
    HMP_FORALL_INTEGRAL_TYPES(DEFINE_INTEGRAL_CTOR)
#undef DEFINE_INTEGRAL_CTRO

#define DEFINE_FLOATING_CTOR(T, _)                                             \
    Scalar(T v) {                                                              \
        type_ = T_Float;                                                       \
        data_.d = v;                                                           \
    }
    HMP_FORALL_FLOATING_TYPES(DEFINE_FLOATING_CTOR)
#undef DEFINE_FLOATING_CTRO

    Scalar(bool v) {
        type_ = T_Bool;
        data_.i = v;
    }

    template <typename T> T to() const {
        if (type_ == T_Float) {
            return static_cast<T>(data_.d);
        } else {
            return static_cast<T>(data_.i);
        }
    }

    bool is_integral(bool include_bool) {
        return type_ == T_Bool || type_ == T_Int;
    }

    bool is_floating_point() { return type_ == T_Float; }

    bool is_boolean() { return type_ == T_Bool; }

  private:
    enum Type { T_Bool, T_Int, T_Float } type_;

    union {
        int64_t i;
        double d;
    } data_;
};

} // namespace hmp
