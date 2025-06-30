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

#include <type_traits>

// FIXME: do not include this file in header files

#include <fmt/format.h>

namespace hmp {

template <typename T> void stringfy(const T &);

template <typename T> struct hasStringfy {
    using ret_type = decltype(hmp::stringfy(std::declval<T>()));
    static constexpr bool value = !std::is_same<ret_type, void>::value;
};

template <> struct hasStringfy<void> {
    using ret_type = void;
    static constexpr bool value = false;
};

} // namespace hmp

namespace fmt {

#if FMT_VERSION < 110000
template <typename T>
using buffered_context = buffer_context<T>;
#endif

template <typename T, typename Char>
struct formatter<T, Char, enable_if_t<hmp::hasStringfy<T>::value>> : formatter<basic_string_view<Char>> {
    auto format(const T &c, buffered_context<Char> &ctx) const {
        return formatter<basic_string_view<Char>>::format(hmp::stringfy(c), ctx);
    }
};

template <typename T>
struct formatter<T, std::enable_if_t<std::is_enum_v<T> && !hmp::hasStringfy<T>::value, char>> :
    formatter<int> {
    auto format(const T& a, format_context& ctx) const {
        return formatter<int>::format((int)a, ctx);
    }
};

} // namespace fmt