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
#include <hmp/tensor.h>
#include <hmp/format.h>


#ifdef HMP_ENABLE_CUDA

TEST(TestTensorOptions, Construct)
{
    using namespace hmp;

    auto opt0 = TensorOptions::make(kCUDA, kFloat64, true);
    auto opt1 = TensorOptions::make(true, kFloat64, kCUDA);
    auto opt2 = TensorOptions::make(true, kFloat64, Device("cuda:0"));
    auto opt3 = TensorOptions::make(kFloat64, kCPU, "cuda:0", true);

    EXPECT_EQ(opt0.dtype(), kFloat64);
    EXPECT_EQ(opt0.device(), Device("cuda:0"));
    EXPECT_EQ(opt0.pinned_memory(), true);

    EXPECT_EQ(opt1.dtype(), kFloat64);
    EXPECT_EQ(opt1.device(), Device("cuda:0"));
    EXPECT_EQ(opt1.pinned_memory(), true);

    EXPECT_EQ(opt2.dtype(), kFloat64);
    EXPECT_TRUE(opt2.device() == Device("cuda:0"));
    EXPECT_EQ(opt2.pinned_memory(), true);

    EXPECT_EQ(opt3.dtype(), kFloat64);
    EXPECT_TRUE(opt3.device() == Device("cuda:0"));
    EXPECT_EQ(opt3.pinned_memory(), true);
}

#endif