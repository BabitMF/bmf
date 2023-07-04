#include <bmf/sdk/media_description.h>
#include <gtest/gtest.h>

using namespace bmf_sdk;

TEST(media_description, construct)
{
    MediaDesc dp;
    dp.width(1920).height(1080).device(hmp::Device("cpu")).pixel_format(hmp::PF_YUV420P);

    EXPECT_EQ(dp.width(), 1920);
    EXPECT_EQ(dp.height(), 1080);
    EXPECT_EQ(dp.device().type(), hmp::Device::Type::CPU);
    EXPECT_EQ(dp.pixel_format(), hmp::PF_YUV420P);
}
