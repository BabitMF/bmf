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

#import "BmfLiteDemoVideoFrame.h"

BMFLITE_DEMO_NAMESPACE_BEGIN

void VideoFrame::setCVPixelBufferRef(CVPixelBufferRef buffer) {
    if (buffer_) {
        CVPixelBufferRelease(buffer_);
        buffer_ = nullptr;
    }
    buffer_ = CVPixelBufferRetain(buffer);
}

void VideoFrame::holdSource() {
    compare_ = true;
    source_ = CVPixelBufferRetain(buffer_);
}

VideoFrame::~VideoFrame() {
    if (buffer_) {
        CVPixelBufferRelease(buffer_);
        buffer_ = nullptr;
    }

    if (nil != sample_buffer_ref_) {
        CFRelease(sample_buffer_ref_);
        sample_buffer_ref_ = nil;
    }
    if (compare_) {
        CVPixelBufferRelease(source_);
        source_ = nullptr;
    }
}


BMFLITE_DEMO_NAMESPACE_END
