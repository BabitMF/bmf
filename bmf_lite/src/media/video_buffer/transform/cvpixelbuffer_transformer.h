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

#ifndef _BMFLITE_CVPIXELBUFFER_TRANSFORMER_H_
#define _BMFLITE_CVPIXELBUFFER_TRANSFORMER_H_

#include "media/video_buffer/transformer.h"
#include <AVFoundation/AVFoundation.h>
#include <Metal/Metal.h>

namespace bmf_lite {

class CvPixelBufferTransformerImpl;
class CvPixelBufferTransformer {
  public:
    CvPixelBufferTransformer();
    ~CvPixelBufferTransformer();

    int init(HardwareDataInfo hardware_data_info_in,
             std::shared_ptr<HWDeviceContext> context,
             HardwareDataInfo hardware_data_info_out);

    int trans(std::shared_ptr<VideoBuffer> in_video_buffer,
              std::shared_ptr<VideoBuffer> &output_video_buffer);

  private:
    std::shared_ptr<CvPixelBufferTransformerImpl> impl_;
};

class PixelBufferAndTxtureFmt {
  public:
    explicit PixelBufferAndTxtureFmt(OSType pixel_format);

    inline MTLPixelFormat getTexFormatByPlane(int plane);

    inline int getPlaneCount();
    inline int getWidthByPlaneIndexWithOriginWidth(int index, int width);
    inline int getHeightByPlaneIndexWithOriginHeight(int index, int height);

    inline bool support();

  private:
    OSType pixel_buffer_fmt_;
    MTLPixelFormat tex0_fmt_;
    MTLPixelFormat tex1_fmt_;
    MTLPixelFormat tex2_fmt_;
    int plane_count_;
    bool support_ = true;
    int32_t plane_ratio_[3];
}; // end class PixelBufferAndTxtureFmt

} // namespace bmf_lite

#endif // _BMFLITE_CVPIXELBUFFER_TRANSFORMER_H_