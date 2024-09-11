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

#ifndef _BMFLITE_FORMAT_H_
#define _BMFLITE_FORMAT_H_

namespace bmf_lite {

enum class MemoryType {
    kUnknown = 0,

    // byte array memory in CPU
    kByteMemory,

    // CVPixelBufferRef for iOS and macOS
    kCVPixelBuffer,

    // OpenGL texture 2d
    kOpenGLTexture2d,

    // OpenGL texture external_oes
    kOpenGLTextureExternalOes,

    // direct3d 11 texture
    kD3D11Texture,

    // direct3d 9 texture
    kD3D9Texture,

    // generic texture
    kGeneralTexture,

    // cuda buffer
    kCuda,

    // vaapi buffer
    kVaapi,

    // ByteAudioFrmae
    kByteAudioFrame,

    // metal Texture
    kMetalTexture,

    // multi Texture
    kMultiMetalTexture,

    kCVPixelBufferAndMetalTexture,

    // float array memory in CPU
    kFloatMemory,

    kRaw,
};

enum GL_INTERNAL_FORMAT {
    GLES_TEXTURE_RGBA = 0,
    GLES_TEXTURE_RGBA8UI = 1,
};

enum CPU_MEMORY_INTERNAL_FORMAT {
    CPU_RGB = 0,
    CPU_RGBA = 1,
    CPU_RGBFLOAT = 2,
    CPU_RGBAFLOAT = 3,
};

enum CVPiexelBuffer_FORMAT {
    BMF_LITE_CV_RGBA = 0,
    BMF_LITE_CV_NV12 = 1,
};

enum MetalTexture_FORMAT {
    BMF_LITE_MTL_RGBA = 0,
    BMF_LITE_MTL_NV12 = 1,
    BMF_LITE_MTL_R8Unorm = 2,
    BMF_LITE_MTL_RG8Unorm = 3,
};

// enum MultiMetalTexture_FORMAT{
//     BMF_LITE_MTL_RGBA = 0,
//     BMF_LITE_MTL_NV12 = 1,
// };

enum CGImage_FORMAT {
    BMF_LITE_CGImage_NONE = 0,
};

class HardwareDataInfo {
  public:
    MemoryType mem_type;

    int internal_format;

    // GL_Texture
    int mutable_flag;

    // Metal Texture
    int storage_mode;
    int usage;

    bool operator==(const HardwareDataInfo &other) {
        if (mem_type == other.mem_type &&
            internal_format == other.internal_format &&
            mutable_flag == other.mutable_flag) {
            return true;
        }
        return false;
    }
};

} // namespace bmf_lite

#endif // _BMFLITE_FORMAT_H_