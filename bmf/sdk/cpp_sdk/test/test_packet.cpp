
#include <memory>
#include <gtest/gtest.h>
#include <bmf/sdk/bmf_type_info.h>
#include <bmf/sdk/packet.h>
#include <bmf/sdk/audio_frame.h>
#include <bmf/sdk/video_frame.h>

using namespace bmf_sdk;

namespace {

struct A{
};

struct B{
};

namespace test{

struct C{ //copyable, moveable
    int value = 0;
};

struct D : public C{ //moveable only
    D(D&&) = default;

    std::unique_ptr<int> v_ptr;

    int *u_ptr = nullptr;

    ~D()
    {
        if(u_ptr){
            *u_ptr = 0x42;
        }
    }
};

} //namespace test

} //namespace

//register in global namespace
BMF_DEFINE_TYPE(test::C)
BMF_DEFINE_TYPE(test::D)

TEST(type_info, type_info)
{
    //default using type_id
    EXPECT_TRUE(type_info<A>() == type_info<const A>());
    EXPECT_TRUE(type_info<A>() == type_info<const A&>());
    EXPECT_TRUE(type_info<A>() == type_info<const A&&>());
    EXPECT_TRUE(type_info<A>() == type_info<A&&>());
    EXPECT_TRUE(type_info<A>() != type_info<B>());

    //register by BMF_DEFINE_TYPE
    auto &c_info = type_info<test::C>();
    auto &d_info = type_info<test::D>();
    EXPECT_TRUE(type_info<A>() != c_info);
    EXPECT_TRUE(type_info<test::D>() != c_info);
    EXPECT_EQ(c_info.name, std::string("test::C"));
    EXPECT_EQ(d_info.name, std::string("test::D"));

    //pre-define types
    EXPECT_EQ(type_info<std::string>().name, std::string("std::string"));
    EXPECT_EQ(type_info<Tensor>().name, std::string("hmp::Tensor"));
    EXPECT_EQ(type_info<VideoFrame>().name, std::string("bmf_sdk::VideoFrame"));
    EXPECT_EQ(type_info<AudioFrame>().name, std::string("bmf_sdk::AudioFrame"));
}




TEST(packet, constructors)
{
    Packet pkt0;
    EXPECT_FALSE(pkt0);

    test::C c{1};
    test::D d{2};

    int d_destroy_flag = 0x0;
    {
        //type cast
        auto pkt_c = Packet(c); //copy c
        Packet pkt_cc = pkt_c;
        //auto pkt_d = Packet(d);  //compile error, as d is not copyable
        auto pkt_d = Packet(std::move(d));

        ASSERT_EQ(pkt_c.unsafe_self(), pkt_cc.unsafe_self());
        EXPECT_EQ(pkt_c.get<test::C>().value, pkt_cc.get<test::C>().value);

        ASSERT_NO_THROW(pkt_c.get<test::C>());
        ASSERT_NO_THROW(pkt_d.get<test::D>());
        ASSERT_THROW(pkt_c.get<test::D>(), std::bad_cast);
        ASSERT_THROW(pkt_d.get<test::C>(), std::bad_cast);

        EXPECT_EQ(pkt_c.get<test::C>().value, 1);
        EXPECT_EQ(pkt_d.get<test::D>().value, 2);

        EXPECT_TRUE(pkt_c.is<test::C>());
        EXPECT_FALSE(pkt_c.is<test::D>());
        EXPECT_TRUE(pkt_d.is<test::D>());
        EXPECT_FALSE(pkt_d.is<test::C>());

        //timestamp
        EXPECT_EQ(pkt_c.timestamp(), UNSET);
        pkt_c.set_timestamp(111);
        EXPECT_EQ(pkt_c.timestamp(), 111);

        pkt_d.get<test::D>().u_ptr = &d_destroy_flag;
        EXPECT_EQ(d_destroy_flag, 0);
    }
    EXPECT_EQ(d_destroy_flag, 0x42); //check if d is destroyed

    auto pkt_eos = Packet::generate_eos_packet();
    EXPECT_EQ(pkt_eos.timestamp(), EOS);

    auto pkt_eof = Packet::generate_eof_packet();
    EXPECT_EQ(pkt_eof.timestamp(), BMF_EOF);

    //
    EXPECT_NO_THROW(Packet(VideoFrame::make(720, 1280, 3, kNCHW)));

}