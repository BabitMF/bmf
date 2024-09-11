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

#ifndef _BMFLITE_GL_TEXTURE_TRANSFORMER_H_
#define _BMFLITE_GL_TEXTURE_TRANSFORMER_H_

#include "media/video_buffer/transformer.h"

namespace bmf_lite {

class GLTextureTransformerImpl;
class GLTextureTransformer {
  public:
    GLTextureTransformer();
    ~GLTextureTransformer();

    int init(HardwareDataInfo hardware_data_info_in,
             std::shared_ptr<HWDeviceContext> context);

    int transTexture2Memory(std::shared_ptr<VideoBuffer> in_video_buffer,
                            std::shared_ptr<VideoBuffer> &out_video_buffer);

    int transMemory2Texture(std::shared_ptr<VideoBuffer> in_video_buffer,
                            std::shared_ptr<VideoBuffer> out_video_buffer);

  private:
    std::shared_ptr<GLTextureTransformerImpl> impl_;
};

} // namespace bmf_lite

#endif // _BMFLITE_GLTEXTURE_TRANSFORMER_H_