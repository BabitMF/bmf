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
#include <gtest/gtest.h>
#include <hmp/core/scalar.h>

namespace hmp{


template<typename T>
void test_scalar_precision(int64_t value)
{
    T v = *reinterpret_cast<T*>(&value);
    auto scalar = Scalar(v);

    auto cast_v = scalar.to<T>();
    EXPECT_EQ(v, cast_v);
}

TEST(TestScalar, precision_test)
{
    int64_t magic_value = 0xdeadbeefdeadbeefll;


#define DISPATCH(dtype, _) { \
        test_scalar_precision<dtype>(magic_value); \
    }

    HMP_FORALL_SCALAR_TYPES(DISPATCH)

#undef DISPATCH
}




} //namespace

