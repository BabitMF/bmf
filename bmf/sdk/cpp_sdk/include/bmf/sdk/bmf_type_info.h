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

#include <stdint.h>
#include <type_traits>
#include <typeinfo>
#include <bmf/sdk/common.h>

namespace bmf_sdk {

// to avoid design a sophistic type system, like boost::type_index,
// we provide template interface for user to register their types

struct TypeInfo {
    const char *name;
    std::size_t index; // unique
};

static inline bool operator==(const TypeInfo &lhs, const TypeInfo &rhs) {
    return lhs.index == rhs.index;
}

static inline bool operator!=(const TypeInfo &lhs, const TypeInfo &rhs) {
    return lhs.index != rhs.index;
}

BMF_API std::size_t string_hash(const char *str);

//
template <typename T> struct TypeTraits {
    static const char *name() { return typeid(T).name(); }
};

template <typename T> const TypeInfo &_type_info() {
    static TypeInfo s_type_info{TypeTraits<T>::name(),
                                string_hash(TypeTraits<T>::name())};
    return s_type_info;
}

template <typename T> const TypeInfo &type_info() {
    using U = std::remove_reference_t<std::remove_cv_t<T>>;
    return _type_info<U>();
}

// Note: use it in global namespace
#define BMF_DEFINE_TYPE_N(T, Name)                                             \
    namespace bmf_sdk {                                                        \
    template <> struct TypeTraits<T> {                                         \
        static const char *name() { return Name; }                             \
    };                                                                         \
    }

#define BMF_DEFINE_TYPE(T) BMF_DEFINE_TYPE_N(T, #T)

} // namespace bmf_sdk

// std types
// BMF_DEFINE_TYPE(std::string)