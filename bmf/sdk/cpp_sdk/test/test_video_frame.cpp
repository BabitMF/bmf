
#include <bmf/sdk/video_frame.h>
#include <bmf/sdk/json_param.h>
#include <bmf/sdk/module_functor.h>
#ifdef BMF_ENABLE_FFMPEG
#include <bmf/sdk/ffmpeg_helper.h>
#endif
#include <bmf/sdk/exception_factory.h>
#include <hmp/core/stream.h>
#include <gtest/gtest.h>


using namespace bmf_sdk;

//NOTE: all correctness of tensor operations are tested in hml/tests

template<typename T>
static bool check_pixel_value(const VideoFrame &vf, const T &v)
{
    //ASSERT_TRUE(vf.is_image());
    //ASSERT_TRUE(vf.device() == kCPU);

    auto image = vf.image().to(kNCHW); //convert to known layout
    auto &data = image.data();
    const T* ptr = data.data<T>();

    for (int64_t c = 0; c < image.nchannels(); ++c){
        for (int64_t y = 0; y < image.height(); ++y){
            for (int64_t x = 0; x < image.width(); ++x){
                auto idx = c * data.stride(0) + y * data.stride(1) + x * data.stride(2);
                if(ptr[idx] != v){
                    return false;
                }
            }
        }
    }

    return true;
}


static VideoFrame decode_one_frame(const std::string &path)
{
    JsonParam option;
    option.parse(fmt::format("{{\"input_path\": \"{}\"}}", path));
    auto decoder = make_sync_func<std::tuple<>, std::tuple<VideoFrame>>(
                                ModuleInfo("c_ffmpeg_decoder"), option);

    VideoFrame vf;
    std::tie(vf) = decoder();
    return vf;
}



TEST(video_frame, frame_constructors)
{
    int width = 1920, height = 1080;

    // default constructor
    VideoFrame undefined; //
    EXPECT_FALSE(undefined);

    //create with default TensorOptions
    auto H420 = PixelInfo(hmp::PF_YUV420P, hmp::CS_BT709);
    auto vf0 = VideoFrame::make(width, height, H420); //
    EXPECT_EQ(vf0.width(), width);
    EXPECT_EQ(vf0.height(), height);
    EXPECT_EQ(vf0.dtype(), kUInt8);
    EXPECT_TRUE(vf0.device() == kCPU);
    EXPECT_FALSE(vf0.is_image());
    EXPECT_NO_THROW(vf0.frame());
    EXPECT_THROW(vf0.image(), std::runtime_error);
    EXPECT_EQ(vf0.frame().format(), hmp::PF_YUV420P);

    // With dtype specified, not support

#ifdef HMP_ENABLE_CUDA
    //with device specifed
    auto vf2 = VideoFrame::make(1920, 1080, H420,
                                "cuda:0"   //cuda device 
                                );
    EXPECT_TRUE(vf2.device() == Device(kCUDA, 0));
    EXPECT_TRUE(vf2.dtype() == kUInt8);

    auto vf3 = VideoFrame::make(1920, 1080, H420,
                                Device(kCUDA, 0)   //cuda device 
                                );
    EXPECT_TRUE(vf3.device() == Device(kCUDA, 0));

    auto vf4 = VideoFrame::make(1920, 1080, H420,
                                kCUDA   //cuda device 
                                );
    EXPECT_TRUE(vf4.device() == Device(kCUDA, 0));

#endif

}



TEST(video_frame, image_constructors)
{
    int width = 1920, height = 1080;

    //create with default TensorOptions
    auto vf0 = VideoFrame::make(width, height); //
    EXPECT_EQ(vf0.width(), width);
    EXPECT_EQ(vf0.height(), height);
    EXPECT_EQ(vf0.dtype(), kUInt8);
    EXPECT_TRUE(vf0.device() == kCPU);
    EXPECT_TRUE(vf0.is_image());
    EXPECT_THROW(vf0.frame(), std::runtime_error);
    EXPECT_NO_THROW(vf0.image());
    EXPECT_EQ(vf0.image().format(), kNCHW);
    EXPECT_EQ(vf0.image().nchannels(), 3);

    // With dtype, channels, ChannelFormat specifed
    auto vf1 = VideoFrame::make(1920, 1080, 1, kNHWC, kUInt16); //
    EXPECT_EQ(vf1.dtype(), kUInt16);
    EXPECT_EQ(vf1.image().nchannels(), 1);
    EXPECT_EQ(vf1.image().format(), kNHWC);

#ifdef HMP_ENABLE_CUDA
    //with type and device specifed
    auto vf2 = VideoFrame::make(1920, 1080, 3, kNHWC,
                                kHalf,
                                "cuda:0"   //cuda device 
                                );
    EXPECT_TRUE(vf2.device() == Device(kCUDA, 0));
    EXPECT_TRUE(vf2.dtype() == kHalf);
    EXPECT_EQ(vf2.image().format(), kNHWC);

    auto vf3 = VideoFrame::make(1920, 1080, 3, kNHWC,
                                kFloat32,
                                true    //pinned_memory, cudaMallocHost
                                ); 
    EXPECT_TRUE(vf3.device() == Device(kCPU));
    EXPECT_TRUE(vf3.image().data().options().pinned_memory());
    EXPECT_EQ(vf3.image().format(), kNHWC);
    EXPECT_EQ(vf3.dtype(), kFloat32);
#endif

}


TEST(video_frame, crop_test)
{
    int width = 1920, height = 1080;

    //frame
    auto H420 = PixelInfo(hmp::PF_YUV420P, hmp::CS_BT709);
    auto vf0 = VideoFrame::make(width, height, H420); //
    auto vf0_sub = vf0.crop(50, 100, 1280, 720);
    EXPECT_EQ(vf0_sub.width(), 1280);
    EXPECT_EQ(vf0_sub.height(), 720);
    EXPECT_FALSE(vf0_sub.is_image());
    auto &vf0_sub_data = vf0_sub.frame().plane(0);
    EXPECT_EQ(vf0_sub_data.stride(0), 1920); //(H, W, 1) layout, stride(0) == line width

    //image
    auto vf1 = VideoFrame::make(width, height, 3, kNCHW); //
    auto vf1_sub = vf1.crop(50, 100, 1280, 720);
    EXPECT_EQ(vf1_sub.width(), 1280);
    EXPECT_EQ(vf1_sub.height(), 720);
    EXPECT_TRUE(vf1_sub.is_image());
    auto &vf1_sub_data = vf1_sub.image().data();
    EXPECT_EQ(vf1_sub_data.stride(1), 1920); //(C, H, W) layout, stride(2) == line width
}


#ifdef HMP_ENABLE_CUDA

TEST(video_frame, copy_test)
{
    int width = 1920, height = 1080;

    //frame
    auto H420 = PixelInfo(hmp::PF_YUV420P, hmp::CS_BT709);
    auto vf0 = VideoFrame::make(width, height, H420); //
    EXPECT_TRUE(vf0.device() == kCPU);
    auto vf0_cuda = vf0.cuda();
    EXPECT_TRUE(vf0_cuda.device() == kCUDA);
    auto vf0_cpu = vf0_cuda.cpu();
    EXPECT_TRUE(vf0_cpu.device() == kCPU);


    //image
    auto vf1 = VideoFrame::make(width, height); //
    EXPECT_TRUE(vf1.device() == kCPU);
    auto vf1_cuda = vf0.cuda();
    EXPECT_TRUE(vf1_cuda.device() == kCUDA);
    auto vf1_cpu = vf0_cuda.cpu();
    EXPECT_TRUE(vf1_cpu.device() == kCPU);

    //inplace copy, support both frame and image
    auto vf3 = VideoFrame::make(width, height, 3, kNCHW, kCUDA); //
    auto vf4 = VideoFrame::make(width+100, height+200, 3, kNCHW, kCUDA); //
    auto vf4_sub = vf4.crop(100, 200, width, height);
    EXPECT_NO_THROW(vf4_sub.copy_(vf3));

}


TEST(video_frame, async_execution)
{
    int width = 1920, height = 1080;
    auto vf0 = VideoFrame::make(width, height, 3, kNCHW, kCPU, true); //allocate pinned memory
    auto vf1 = VideoFrame::make(width, height, 3, kNCHW, kCUDA); 
    VideoFrame vf2;

    //
    auto data = vf0.image().data(); //shadow copy, remove const
    data.fill_(1); //(1, 1, 1, 1, .....)

    //with cuda stream support
    auto stream = hmp::create_stream(kCUDA);
    {
        hmp::StreamGuard guard(stream);
        EXPECT_EQ(stream, hmp::current_stream(kCUDA));

        vf1.copy_(vf0);

        auto data = vf1.image().data();
        data += 2; // (3, 3, 3, 3, ....)

        vf2 = vf1.cpu(true); //async copy to cpu 

        vf2.record();
        EXPECT_FALSE(vf2.ready());

        vf2.synchronize();
        //stream.synchronize();

        EXPECT_TRUE(vf2.ready());
    }

    ASSERT_EQ(vf2.dtype(), kUInt8);
    EXPECT_TRUE(check_pixel_value<uint8_t>(vf2, 3));
}

#endif


namespace bmf_sdk{

struct MockAVFrame
{
    MockAVFrame(bool *valid_p) : valid(valid_p)
    {
        *valid = true;
    }

    ~MockAVFrame()
    {
        if(valid){
            *valid = false;
        }
    }

    int value = 0;

    bool *valid = nullptr;
};

template<>
struct OpaqueDataInfo<MockAVFrame>
{
    const static int key = OpaqueDataKey::kAVFrame;

    static OpaqueData construct(const MockAVFrame *avf)
    {
        return OpaqueData(const_cast<MockAVFrame*>(avf), [](void *p){
            delete (MockAVFrame*)p;
        });
    }
};

} //namespace bmf_sdk

TEST(video_frame, private_data)
{
    int width = 1920, height = 1080;
    bool valid = false;

    {
        auto pri_data = new MockAVFrame(&valid);
        pri_data->value = 42;
        auto vf0 = VideoFrame::make(width, height, 3, kNCHW, kCPU, false);
        vf0.private_attach(pri_data); // vf0 will own pri_data
        EXPECT_EQ(vf0.private_get<MockAVFrame>()->value, 42);
        EXPECT_EQ(valid, true);

        //
        auto vf1 = VideoFrame::make(width, height, 3, kNCHW, kCPU, false); 
        vf1.copy_(vf0);
        vf1.private_merge(vf0); // now, vf0 and vf1 will share the same private data
        EXPECT_EQ(vf1.private_get<MockAVFrame>()->value, 42);

        //vf1.private_get<MockAVFrame>()->value = 100; //modify already set private data is not allowed
                                                       //as it may cause unpredictable error(it may used in other modules)
        //solution to modify private data is to copy it
        auto pri_data_copy = new MockAVFrame(*vf0.private_get<MockAVFrame>()); //copy
        EXPECT_EQ(pri_data_copy->value, 42);
        pri_data_copy->valid = nullptr;
        pri_data_copy->value = 100;
        vf1.private_attach(pri_data_copy);
        EXPECT_EQ(vf1.private_get<MockAVFrame>()->value, 100);
        EXPECT_EQ(vf0.private_get<MockAVFrame>()->value, 42);

        //
        EXPECT_TRUE(valid); //ensure pri_data is alive
    }

    EXPECT_FALSE(valid); //pri_data is destructed
}


TEST(video_frame, private_data_json_param)
{
    auto vf = VideoFrame::make(1920, 1080, 3, kNCHW, kCPU, false); 
    auto json_sptr = vf.private_get<JsonParam>();
    EXPECT_FALSE(json_sptr);

    JsonParam ref; 
    ref.parse("{\"v\": 42}");
    vf.private_attach(&ref); //copy it internally

    auto data_sptr = vf.private_get<JsonParam>();
    ASSERT_TRUE(data_sptr);

    EXPECT_EQ(data_sptr->get<int>("v"), 42);
}




TEST(video_frame, image_frame_convert)
{
    auto H420 = PixelInfo(hmp::PF_YUV420P, hmp::CS_BT709);
    int width = 1920, height = 1080;
    auto vf0 = VideoFrame::make(width, height, 3, kNCHW, kCPU, false); //Image type
    auto vf1 = VideoFrame::make(width, height, H420); //Frame type

    //Image layout convert
    auto vf2 = vf0.to_image(kNHWC); 
    EXPECT_TRUE(vf2.is_image());
    EXPECT_EQ(vf2.image().format(), kNHWC);

    //Convert Image to Frame
    auto vf3 = vf0.to_frame(H420); 
    EXPECT_FALSE(vf3.is_image());
    EXPECT_EQ(vf3.frame().format(), hmp::PF_YUV420P);

    //Convert Frame to Image with layout NCHW
    auto vf4 = vf1.to_image(kNCHW);
    EXPECT_TRUE(vf4.is_image());
    EXPECT_EQ(vf4.image().format(), kNCHW);
}


TEST(video_frame, copy_props)
{
    auto H420 = PixelInfo(hmp::PF_YUV420P, hmp::CS_BT709);
    int width = 1920, height = 1080;
    auto vf0 = VideoFrame::make(width, height, 3, kNCHW, kCPU, false); //Image type
    auto vf1 = VideoFrame::make(width, height, H420); //Frame type

    vf0.set_stream(42);
    vf0.set_time_base(Rational(1, 2));
    vf0.set_pts(100);

    vf1.copy_props(vf0);
    EXPECT_EQ(vf1.stream(), 42);
    EXPECT_EQ(vf1.pts(), 100);
    EXPECT_EQ(vf1.time_base().den, 2);
    EXPECT_EQ(vf1.time_base().num, 1);
}

#ifdef BMF_ENABLE_FFMPEG
TEST(video_frame, reformat)
{
    auto ori_vf = decode_one_frame("../files/img.mp4");
    ASSERT_FALSE(ori_vf.is_image());
    ASSERT_EQ(ori_vf.frame().format(), hmp::PF_YUV420P);
    EXPECT_EQ(ori_vf.height(), 1080);
    EXPECT_EQ(ori_vf.width(), 1920);
    ASSERT_FALSE(ori_vf.frame().pix_info().is_rgbx());
    EXPECT_EQ(ori_vf.frame().nplanes(), 3);
    EXPECT_EQ(ori_vf.frame().plane(0).stride(0), 1920);
    EXPECT_EQ(ori_vf.frame().plane(1).stride(0), 1920 / 2);
    EXPECT_EQ(ori_vf.frame().plane(2).stride(0), 1920 / 2);

    //reformat yuv420p -> rgb
    {
        auto rgb_vf = ffmpeg::reformat(ori_vf, "rgb24");
        EXPECT_EQ(rgb_vf.height(), 1080);
        EXPECT_EQ(rgb_vf.width(), 1920);
        ASSERT_FALSE(rgb_vf.is_image());
        EXPECT_EQ(rgb_vf.frame().nplanes(), 1);
        EXPECT_EQ(rgb_vf.frame().height(), 1080);
        EXPECT_EQ(rgb_vf.frame().width(), 1920);
        ASSERT_EQ(rgb_vf.frame().format(), hmp::PF_RGB24);
        ASSERT_TRUE(rgb_vf.frame().pix_info().is_rgbx());
        EXPECT_EQ(rgb_vf.frame().plane(0).stride(0), 3 * 1920);
    }

    auto img_vf = ori_vf.to_image(kNHWC);
    EXPECT_EQ(img_vf.height(), 1080);
    EXPECT_EQ(img_vf.width(), 1920);
    EXPECT_TRUE(img_vf.is_image());
    EXPECT_EQ(img_vf.image().height(), 1080);
    EXPECT_EQ(img_vf.image().width(), 1920);
    EXPECT_EQ(img_vf.image().nchannels(), 3);

    // Image reformat(gray)
    {
        auto gray_vf = ffmpeg::reformat(img_vf, "gray");
        EXPECT_EQ(gray_vf.height(), 1080);
        EXPECT_EQ(gray_vf.width(), 1920);
        ASSERT_FALSE(gray_vf.is_image());
        EXPECT_EQ(gray_vf.frame().nplanes(), 1);
        EXPECT_EQ(gray_vf.frame().height(), 1080);
        EXPECT_EQ(gray_vf.frame().width(), 1920);
        ASSERT_EQ(gray_vf.frame().format(), hmp::PF_GRAY8);
        ASSERT_TRUE(gray_vf.frame().pix_info().is_rgbx());
        EXPECT_EQ(gray_vf.frame().plane(0).stride(0), 1920);
    }

    // Image reformat
    {
        auto gray_vf = ffmpeg::reformat(img_vf, "gray16");
        EXPECT_EQ(gray_vf.height(), 1080);
        EXPECT_EQ(gray_vf.width(), 1920);
        ASSERT_FALSE(gray_vf.is_image());
        EXPECT_EQ(gray_vf.frame().nplanes(), 1);
        EXPECT_EQ(gray_vf.frame().height(), 1080);
        EXPECT_EQ(gray_vf.frame().width(), 1920);
        ASSERT_EQ(gray_vf.frame().format(), hmp::PF_GRAY16);
        ASSERT_TRUE(gray_vf.frame().pix_info().is_rgbx());
        EXPECT_EQ(gray_vf.frame().plane(0).stride(0), 1920);
    }

    {
        auto rgb_vf = ffmpeg::reformat(img_vf, "rgb24");
        EXPECT_EQ(rgb_vf.height(), 1080);
        EXPECT_EQ(rgb_vf.width(), 1920);
        ASSERT_FALSE(rgb_vf.is_image());
        EXPECT_EQ(rgb_vf.frame().nplanes(), 1);
        EXPECT_EQ(rgb_vf.frame().height(), 1080);
        EXPECT_EQ(rgb_vf.frame().width(), 1920);
        ASSERT_EQ(rgb_vf.frame().format(), hmp::PF_RGB24);
        ASSERT_TRUE(rgb_vf.frame().pix_info().is_rgbx());
        EXPECT_EQ(rgb_vf.frame().plane(0).stride(0), 3 * 1920);
    }

    {
        auto yuv_vf = ffmpeg::reformat(img_vf, "yuv420p");
        EXPECT_EQ(yuv_vf.height(), 1080);
        EXPECT_EQ(yuv_vf.width(), 1920);
        ASSERT_FALSE(yuv_vf.is_image());
        EXPECT_EQ(yuv_vf.frame().nplanes(), 3);
        EXPECT_EQ(yuv_vf.frame().height(), 1080);
        EXPECT_EQ(yuv_vf.frame().width(), 1920);
        ASSERT_EQ(yuv_vf.frame().format(), hmp::PF_YUV420P);
        ASSERT_FALSE(yuv_vf.frame().pix_info().is_rgbx());
        EXPECT_EQ(yuv_vf.frame().plane(0).stride(0), 1920);
        EXPECT_EQ(yuv_vf.frame().plane(1).stride(0), 1920 / 2);
        EXPECT_EQ(yuv_vf.frame().plane(2).stride(0), 1920 / 2);
    }

    // Frame reformat
    {
        auto yuv_vf = ffmpeg::reformat(ori_vf, "yuv420p");
        EXPECT_EQ(yuv_vf.height(), 1080);
        EXPECT_EQ(yuv_vf.width(), 1920);
        ASSERT_FALSE(yuv_vf.is_image());
        EXPECT_EQ(yuv_vf.frame().nplanes(), 3);
        EXPECT_EQ(yuv_vf.frame().height(), 1080);
        EXPECT_EQ(yuv_vf.frame().width(), 1920);
        ASSERT_EQ(yuv_vf.frame().format(), hmp::PF_YUV420P);
        ASSERT_FALSE(yuv_vf.frame().pix_info().is_rgbx());
        EXPECT_EQ(yuv_vf.frame().plane(0).stride(0), 1920);
        EXPECT_EQ(yuv_vf.frame().plane(1).stride(0), 1920 / 2);
        EXPECT_EQ(yuv_vf.frame().plane(2).stride(0), 1920 / 2);
    }
}
#endif