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

#include <string>
#include <stdint.h>
#include <hmp/core/half.h>

namespace hmp {

#define HMP_FORALL_SCALAR_TYPES(_)                                             \
    _(uint8_t, UInt8)                                                          \
    _(int8_t, Int8)                                                            \
    _(uint16_t, UInt16)                                                        \
    _(int16_t, Int16)                                                          \
    _(int32_t, Int32)                                                          \
    _(int64_t, Int64)                                                          \
    _(float, Float32)                                                          \
    _(double, Float64)                                                         \
    _(Half, Half)

#define HMP_FORALL_INTEGRAL_TYPES(_)                                           \
    _(uint8_t, UInt8)                                                          \
    _(int8_t, Int8)                                                            \
    _(uint16_t, UInt16)                                                        \
    _(int16_t, Int16)                                                          \
    _(int32_t, Int32)                                                          \
    _(int64_t, Int64)

#define HMP_FORALL_FLOATING_TYPES(_)                                           \
    _(float, Float32)                                                          \
    _(double, Float64)                                                         \
    _(Half, Half)

enum class ScalarType : int8_t {
#define DEFINE_ENUM(_, name) name,
    HMP_FORALL_SCALAR_TYPES(DEFINE_ENUM)
#undef DEFINE_ENUM
        Undefined
};

// helper enums types
#define DEFINE_KSCALAR(_, S) const static ScalarType k##S = ScalarType::S;
HMP_FORALL_SCALAR_TYPES(DEFINE_KSCALAR)
#undef DEFINE_KSCALAR

// Type comversion helpers
namespace impl {

template <ScalarType S> struct ScalarTypeToCPPType;

#define DEFINE_SCALAR_TYPE_TO_TYPE(T, S)                                       \
    template <> struct ScalarTypeToCPPType<ScalarType::S> {                    \
        using type = T;                                                        \
    };
HMP_FORALL_SCALAR_TYPES(DEFINE_SCALAR_TYPE_TO_TYPE)
#undef DEFINE_SCALAR_TYPE_TO_TYPE

template <typename T> struct CPPTypeToScalarType;

#define DEFINE_CPP_TYPE_TO_SCALAR_TYPE(T, S)                                   \
    template <> struct CPPTypeToScalarType<T> {                                \
        static const ScalarType scalar_type = ScalarType::S;                   \
    };
HMP_FORALL_SCALAR_TYPES(DEFINE_CPP_TYPE_TO_SCALAR_TYPE)
#undef DEFINE_CPP_TYPE_TO_SCALAR_TYPE

} // namespace impl

template <ScalarType S>
using getCppType = typename impl::ScalarTypeToCPPType<S>::type;

template <typename T> constexpr ScalarType getScalarType() {
    return impl::CPPTypeToScalarType<T>::scalar_type;
}

constexpr size_t sizeof_scalar_type(ScalarType s) {
    switch (s) {
#define GET_SCALAR_SIZE(T, S)                                                  \
    case ScalarType::S:                                                        \
        return sizeof(T);

        HMP_FORALL_SCALAR_TYPES(GET_SCALAR_SIZE)

#undef GET_SCALAR_SIZE
    default:
        return 0;
        break;
    }
}

// stringfy
static inline std::string stringfy(ScalarType scalar_type) {
    switch (scalar_type) {
#define STRINGFY_SCALAR_TYPE(_, S)                                             \
    case ScalarType::S:                                                        \
        return "k" #S;
        HMP_FORALL_SCALAR_TYPES(STRINGFY_SCALAR_TYPE)
#undef STRINGFY_SCALAR_TYPE
    default:
        return "UnknownScalarType";
    }
}

} // namespace hmp