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

#ifndef BMF_RATIONAL_H
#define BMF_RATIONAL_H

#include <bmf/sdk/common.h>
#include <cstdint>

BEGIN_BMF_SDK_NS
struct BMF_API Rational {
    Rational() {}

    Rational(int n, int d) {
        num = n;
        den = d;
    }

    int num = -1; ///< Numerator
    int den = -1; ///< Denominator
};
END_BMF_SDK_NS

#endif // BMF_RATIONAL_H
