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

#include <bmf/sdk/bmf_av_packet.h>
#include <gtest/gtest.h>

using namespace bmf_sdk;

TEST(bmf_av_packet, constructors) {
    BMFAVPacket pkt0;
    EXPECT_FALSE(pkt0);

    EXPECT_THROW(pkt0 = BMFAVPacket(Tensor()), std::runtime_error);
    auto d0 = hmp::empty({1024}).slice(0, 0, -1, 2); // non-contiguous
    EXPECT_THROW(pkt0 = BMFAVPacket(d0), std::runtime_error);

    auto pkt1 = BMFAVPacket::make(1024);
    EXPECT_EQ(pkt1.data().dtype(), kUInt8);
    EXPECT_EQ(pkt1.data().dim(), 1);
    EXPECT_EQ(pkt1.data().size(0), 1024);
    EXPECT_EQ(pkt1.nbytes(), 1024);
    EXPECT_TRUE(pkt1.data_ptr() != nullptr);
}

TEST(bmf_av_packet, copy_props) {
    auto pkt0 = BMFAVPacket::make(1024);
    pkt0.set_time_base(Rational(1, 2));
    pkt0.set_pts(100);

    auto pkt1 = BMFAVPacket::make(1024);
    pkt1.copy_props(pkt0);

    EXPECT_EQ(pkt1.pts(), 100);
    EXPECT_EQ(pkt1.time_base().den, 2);
    EXPECT_EQ(pkt1.time_base().num, 1);
}