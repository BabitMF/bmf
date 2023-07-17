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

// FIXME: do not include this file in header files

#include <fmt/format.h>

namespace hmp {

template <typename T> void stringfy(const T &);

template <typename T> struct hasStringfy {
    using ret_type = decltype(hmp::stringfy(std::declval<T>()));
    static constexpr bool value = !std::is_same<ret_type, void>::value;
};

} // namespace hmp

template <typename T, typename Char>
struct fmt::formatter<T, Char, fmt::enable_if_t<hmp::hasStringfy<T>::value>> {
    template <typename ParseContext> constexpr auto parse(ParseContext &ctx) {
        return ctx.begin();
    }

    auto format(const T &c, fmt::format_context &ctx) {
        return fmt::format_to(ctx.out(), "{}", hmp::stringfy(c));
    }
};