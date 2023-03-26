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
#ifndef _HMP_OC_METAL_H
#define _HMP_OC_METAL_H

#ifdef __OBJC__
#import <Foundation/Foundation.h>
#import <AVFoundation/AVFoundation.h>
#endif
#include <memory>
#include <hmp/imgproc/formats.h>

namespace hmp{
namespace metal{

// MTLTexture
class Texture
{
    struct Private;
    std::shared_ptr<Private> self;
public:
    enum Usage{
        kWrite = 0x1,
        kRead = 0x2
    };

    Texture() = default;

#ifdef __OBJC__
    Texture(id<MTLTexture> texture);

    id<MTLTexture> texture();
#endif
    Texture(void *texture);

    //creae texture 2d
    Texture(int width, int height, PixelFormat format,
            unsigned usage = kRead | kWrite, bool mipmapped = false);


    const void* handle() const;

    int width() const;
    int height() const;
    int depth() const;
    PixelFormat format() const;
    int texture_type() const; //MTLTextureType
    int pixel_format() const; //MTLPixelFormat
    int sample_count() const; //
    bool read(void *data, int bytesPerRow);
    bool write(const void *data, int bytesPerRow);

    //
    static int max_width();
    static int max_height();
};


unsigned toMTLPixelFormat(PixelFormat format);
PixelFormat fromMTLPixelFormat(unsigned mtlFormat);

class Device
{
private:
    struct Private;
    std::shared_ptr<Private> self;
public:
    Device() = default;

#ifdef __OBJC__
    Device(id<MTLDevice> device);

    id<MTLDevice> device();
#endif
    Device(void *device);

    const void* handle() const;

    static Device& current();
    static void set_current(const Device &dev);
};



}} // namespace

#endif