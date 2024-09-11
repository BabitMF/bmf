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
#ifndef OHOS_SPLIT_SCREEN_RENDERER_H
#define OHOS_SPLIT_SCREEN_RENDERER_H

#include "common/common.h"
#include "shader_program.h"
#include <memory>

namespace bmf_lite_demo {
class SplitScreenRenderer {
  public:
    SplitScreenRenderer();
    ~SplitScreenRenderer();

    int init();
    void setDisplayDivider(int display_divider);
    void setSplitRatio(float ratio);
    void setMatrix(std::vector<float> matrix);
    int drawToScreen(GLuint left_texture, GLuint right_texture, int width,
                     int height);

  private:
    std::unique_ptr<ShaderProgram> textureSplitShaderProgram_;

    GLuint vertexArrayObject_ = GL_NONE;
    GLuint vertexBufferObject_ = GL_NONE;

    GLuint frameBuffer_ = GL_NONE;

    std::vector<float> matrix_;

    float split_ratio_ = 0.5f;
    int display_divider_ = 0;
};
} // namespace bmf_lite_demo

#endif // OHOS_SPLIT_SCREEN_RENDERER_H
