/*
 * Copyright 2024 Babit Authors
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

#ifndef _BMFLITE_DEMO_VIDEO_FRAME_H_
#define _BMFLITE_DEMO_VIDEO_FRAME_H_

#import "BmfliteDemoMacro.h"
#import <AVFoundation/AVFoundation.h>
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

BMFLITE_DEMO_NAMESPACE_BEGIN

class VideoFrame {
  public:
    virtual ~VideoFrame();
    virtual void setCVPixelBufferRef(CVPixelBufferRef buffer);

    CVPixelBufferRef buffer_ = nullptr;
    CMSampleBufferRef sample_buffer_ref_ = nil;
    id<MTLTexture> tex0_ = nil;
    id<MTLTexture> tex1_ = nil;
    id<MTLTexture> tex2_ = nil;

    CMTime duration_;
    CMTime p_time_;
    OSType source_pixel_format_ =
        kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange;
    bool first_ = false;
    bool eos_ = false;

    void holdSource();
    bool compare_ = false;
    CVPixelBufferRef source_ = nil;
    id<MTLTexture> s_tex0_ = nil;
    id<MTLTexture> s_tex1_ = nil;
    id<MTLTexture> s_tex2_ = nil;
}; // end class VideoFrame

BMFLITE_DEMO_NAMESPACE_END

#endif /* _BMFLITE_DEMO_VIDEO_FRAME_H_ */
