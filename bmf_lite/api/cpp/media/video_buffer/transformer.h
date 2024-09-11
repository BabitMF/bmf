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

#ifndef _BMFLITE_MEDIA_TRANSFORMER_H_
#define _BMFLITE_MEDIA_TRANSFORMER_H_

#include "pool.h"
#include "video_buffer.h"

namespace bmf_lite {

class Transformer {
  public:
    Transformer(HardwareDataInfo hardware_data_info_in,
                HardwareDataInfo hardware_data_info_out);
    int trans(VideoBuffer *in_video_buffer, VideoBuffer *output_video_buffer);
};

} // namespace bmf_lite

#endif // _BMFLITE_MEDIA_TRANSFORMER_H_