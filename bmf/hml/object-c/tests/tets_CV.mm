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
#import <test_CV.h>
#import <Metal/Metal.h>


static EAGLContext* CreateBestEAGLContext()
{
   EAGLContext *context = [[EAGLContext alloc] initWithAPI:kEAGLRenderingAPIOpenGLES3];
   if (context == nil) {
      context = [[EAGLContext alloc] initWithAPI:kEAGLRenderingAPIOpenGLES2];
   }
   return context;
}

static int testPixelBuffer()
{
    using namespace hmp;

    @autoreleasepool{ //create MTLTexture from CVPixelBuffer
        id <MTLDevice> device = MTLCreateSystemDefaultDevice();
        metal::Device::set_current(metal::Device(device));

        std::vector<PixelFormat> mtlFormats{
            PF_NV12,
            PF_YUV420P,
            PF_BGRA32
        };

        for(size_t i = 0; i < mtlFormats.size(); ++i){
            oc::PixelBuffer pixelBuffer(1920, 1080, mtlFormats[i], CR_MPEG, false, true);
            PixelFormatDesc desc(mtlFormats[i]);

            for(int j = 0; j < desc.nplanes(); ++j){
                auto tex = pixelBuffer.createMetalTexture(j);
            }
        }
    }

    @autoreleasepool{
        auto glContext = CreateBestEAGLContext();
        if(glContext){
            //https://github.com/sortofsleepy/metalTextureTest/blob/master/src/MetalCam.mm
            //https://stackoverflow.com/questions/27149380/how-to-list-all-opengl-es-compatible-pixelbuffer-formats
            std::vector<PixelFormat> glFormats{
                PF_NV12,
                PF_BGRA32
            };

            for(size_t i = 0; i < glFormats.size(); ++i){
                oc::PixelBuffer pixelBuffer(1920, 1080, glFormats[i], CR_MPEG, true, true);
                PixelFormatDesc desc(glFormats[i]);

                for(int j = 0; j < desc.nplanes(); ++j){
                    auto texId = pixelBuffer.createGlTexture(j, (__bridge void*)glContext);
                }
            }
            
        }
    }

    return 0;
}



@implementation BmfCVTests

- (int) testAll
{
    //
    try{ // 0 - 20
        auto rc = testPixelBuffer();
        if(rc != 0){
            return rc;
        }
    }
    catch(std::exception &e){
        printf("textPixelBuffer failed %s\n", e.what());
        return -19;
    }

    return 0;
}

@end
