#include <vector>
#import <hmp/oc/CV.h>
#import <hmp/oc/Metal.h>
#import <test_Metal.h>
#import <Metal/Metal.h>


static int testMetalTexture()
{
    using namespace hmp;

    @autoreleasepool{ //create MTLTexture from CVPixelBuffer
        id <MTLDevice> device = MTLCreateSystemDefaultDevice();
        metal::Device::set_current(metal::Device(device));

        std::vector<PixelFormat> mtlFormats{
            PF_GRAY8,
            PF_YA8,
            PF_BGRA32,
            PF_RGBA32
        };

        for(size_t i = 0; i < mtlFormats.size(); ++i){
            auto tex = metal::Texture(1920, 1080, mtlFormats[i]);
            if(!tex.handle() || tex.format() != mtlFormats[i] || tex.width() != 1920 || tex.height() != 1080){
                return i+1;
            }
        }
    }

    return 0;
}



@implementation BmfMetalTests

- (int) testAll
{
    //
    try{ // 0 - 20
        auto rc = testMetalTexture();
        if(rc != 0){
            return rc;
        }

        printf("MTLTexture::max_size = %dx%d\n", 
            hmp::metal::Texture::max_width(), hmp::metal::Texture::max_height());

    }
    catch(std::exception &e){
        printf("textMetalTexture failed %s\n", e.what());
        return -19;
    }

    return 0;
}

@end
