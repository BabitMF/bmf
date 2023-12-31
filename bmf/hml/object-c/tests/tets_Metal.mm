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
