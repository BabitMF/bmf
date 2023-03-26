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
#ifndef _HMP_OC_CV_H
#define _HMP_OC_CV_H

#ifdef __OBJC__
#import <Foundation/Foundation.h>
#import <AVFoundation/AVFoundation.h>
#endif
#include <memory>
#include <hmp/imgproc/formats.h>
#include <hmp/oc/Metal.h>


namespace hmp{
namespace oc{

class PixelBuffer
{
    struct Private;
    std::shared_ptr<Private> self;
public:
    PixelBuffer() = default;
#ifdef __OBJC__
    PixelBuffer(CVPixelBufferRef pixel_buffer);

    CVPixelBufferRef buffer();
#endif
    PixelBuffer(void *pixel_buffer);

    PixelBuffer(int width, int height, 
                PixelFormat format, ColorRange range,
                bool gl=false, bool metal=true);

    unsigned createGlTexture(int plane, void *context) const;
#ifdef __OBJC__
    metal::Texture createMetalTexture(int plane);
#endif

    int width() const;
    int height() const;
    int format() const;
    ColorRange range() const;

    const void* handle() const;
}; //


int toCVPixelFormat(PixelFormat format, ColorRange range);
PixelFormat fromCVPixelFormat(int cvFormat);


}} //namespace

#endif