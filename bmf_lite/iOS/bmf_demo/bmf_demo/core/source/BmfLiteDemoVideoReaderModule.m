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

#import "BmfLiteDemoVideoReaderModule.h"
#import "BmfLiteDemoErrorCode.h"
#import "BmfLiteDemoLog.h"

BMFLITE_DEMO_NAMESPACE_BEGIN

VideoReaderModule::VideoReaderModule(NSString* video_path, uint32_t color_pixel_format) {
    video_path_ = [video_path copy];
    color_pixel_format_  = color_pixel_format;
}

int VideoReaderModule::init() {
    name_ = "VideoReaderEffect";
    NSURL *url = [NSURL fileURLWithPath:video_path_];
//    color_pixel_format_ = kCVPixelFormatType_420YpCbCr8BiPlanarFullRange;
//    color_pixel_format_ = kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange;
//    color_pixel_format_ = kCVPixelFormatType_ARGB2101010LEPacked;
    NSDictionary *inputOptions = [NSDictionary
                                  dictionaryWithObject:[NSNumber numberWithBool:true]
                                  forKey:AVURLAssetPreferPreciseDurationAndTimingKey];
    
    input_asset_ = [[AVURLAsset alloc] initWithURL:url options:inputOptions];
    
    [input_asset_
     loadValuesAsynchronouslyForKeys:[NSArray arrayWithObject:@"tracks"]
     completionHandler:^{
        NSError *error = nil;
        AVKeyValueStatus tracksStatus =
        [input_asset_ statusOfValueForKey:@"tracks"
                                  error:&error];
        if (tracksStatus != AVKeyValueStatusLoaded) {
            BMFLITE_DEMO_LOGE(@"BMFModsVideoReaderEffect", @"error %@", error);
            return;
        }
        // processWithAsset(inputAsset);
    }];
    BMFLITE_DEMO_LOGE(@"BMFModsVideoReaderEffect", @"processWithAsset");
    NSError *error = nil;
    asset_reader_ = [AVAssetReader assetReaderWithAsset:input_asset_ error:&error];
    
    NSMutableDictionary *outputSettings = [NSMutableDictionary dictionary];
    
    [outputSettings setObject:@(color_pixel_format_)
                       forKey:(id)kCVPixelBufferPixelFormatTypeKey];
    AVAssetTrack *video_track = [[input_asset_ tracksWithMediaType:AVMediaTypeVideo] objectAtIndex:0] ;
    reader_video_track_output_ = [AVAssetReaderTrackOutput assetReaderTrackOutputWithTrack:video_track
                                                                            outputSettings:outputSettings];
//    reader_video_track_output_.alwaysCopiesSampleData = false;
    reader_video_track_output_.alwaysCopiesSampleData = true;

    fps_ = video_track.nominalFrameRate;
    [asset_reader_ addOutput:reader_video_track_output_];
    duration_ = input_asset_.duration;

    if ([asset_reader_ startReading] == false) {
        BMFLITE_DEMO_LOGE(@"BMFModsVideoReaderEffect", @"Error reading from file at URL: %@", input_asset_);
        return BmfLiteErrorCode::MODULE_INIT_FAILED;
    }

    return BmfLiteErrorCode::SUCCESS;
}

double VideoReaderModule::getFps() {
    return fps_;
}

int VideoReaderModule::process(std::shared_ptr<VideoFrame> data) {
    std::lock_guard<std::mutex> lk(mtx_);
    if (asset_reader_ && asset_reader_.status == AVAssetReaderStatusCompleted) {
        return BmfLiteErrorCode::VIDEO_DECODE_FAILED;
    }
    @autoreleasepool {
        CMSampleBufferRef sampleBufferRef = nil;
        if (reader_video_track_output_) {
            sampleBufferRef = [reader_video_track_output_ copyNextSampleBuffer];
        }
        if (nil == sampleBufferRef) {
            if (!send_eos_) {
                current_video_frame_ = std::make_shared<VideoFrame>();
                current_video_frame_->eos_ = true;
                send_eos_ = true;
                return BmfLiteErrorCode::SUCCESS;
            }
            return BmfLiteErrorCode::VIDEO_DECODE_FAILED;
        }
        
        std::shared_ptr<VideoFrame> frame = std::make_shared<VideoFrame>();
        if (first_) {
            frame->first_ = true;
            frame->source_pixel_format_ = this->color_pixel_format_;
            this->first_ = false;
        }
        frame->duration_ = this->duration_;
        frame->p_time_ = CMSampleBufferGetPresentationTimeStamp(sampleBufferRef);
        CVPixelBufferRef pixelBuffer = CMSampleBufferGetImageBuffer(sampleBufferRef);
        frame->setCVPixelBufferRef(pixelBuffer);
        frame->sample_buffer_ref_ = sampleBufferRef;
        current_video_frame_ = frame;
    }
    return BmfLiteErrorCode::SUCCESS;
}

int VideoReaderModule::close() {
    return BmfLiteErrorCode::FUNCTION_NOT_IMPLEMENT;
}

VideoReaderModule::~VideoReaderModule() {
}

std::shared_ptr<VideoFrame> VideoReaderModule::getCurrentVideoFrame() {
    std::lock_guard<std::mutex> lk(mtx_);
    std::shared_ptr<VideoFrame> frame = current_video_frame_;
    current_video_frame_ = nullptr;
    return frame;
}

BMFLITE_DEMO_NAMESPACE_END
