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

#ifndef _BMFLITE_DEMO_VIDEO_READER_MODULE_H_
#define _BMFLITE_DEMO_VIDEO_READER_MODULE_H_

#import "BmfLiteDemoMacro.h"
#import "BmfLiteDemoVideoFrame.h"
#import "BmfLiteDemoModule.h"
#include <mutex>

BMFLITE_DEMO_NAMESPACE_BEGIN

class VideoReaderModule : public Module {
  public:
    VideoReaderModule(NSString *video_path,
                      uint32_t color_pixel_format =
                          kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange);

    VideoReaderModule();

    virtual ~VideoReaderModule();

    int process(std::shared_ptr<VideoFrame> data) override;

    int init() override;

    int close() override;

    std::shared_ptr<VideoFrame> getCurrentVideoFrame() override;

    double getFps();

  private:
    NSString *video_path_ = nil;
    OSType color_pixel_format_;
    CMTime duration_;
    std::mutex mtx_;
    std::shared_ptr<VideoFrame> current_video_frame_ = nullptr;

    AVAssetReader *asset_reader_;
    AVAssetReaderTrackOutput *reader_video_track_output_;
    AVURLAsset *input_asset_;
    bool first_ = true;
    bool send_eos_ = false;
    bool play_audio_ = true;
    double fps_ = 30;
}; // end class VideoReaderModule

BMFLITE_DEMO_NAMESPACE_END

#endif /* _BMFLITE_DEMO_VIDEO_READER_MODULE_H_ */
