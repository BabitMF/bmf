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
#ifndef OHOS_OES_TO_2D_RENDERER_H
#define OHOS_OES_TO_2D_RENDERER_H

#include <memory>

#include "shader_program.h"
#include "common/common.h"

namespace bmf_lite_demo {

class OesTo2dRenderer {
  public:
    OesTo2dRenderer();
    ~OesTo2dRenderer();

    int init();
    void setMatrix(std::vector<float> matrix);
    int process(GLuint input_texture, GLuint output_texture, int width,
                int height);
    int drawToScreen(GLuint input_texture, int width, int height);

  private:
    std::unique_ptr<ShaderProgram> textureOesShaderProgram_;

    GLuint vertexArrayObject_ = GL_NONE;
    GLuint vertexBufferObject_ = GL_NONE;

    GLuint frameBuffer_ = GL_NONE;

    std::vector<float> matrix_;
};
} // namespace bmf_lite_demo
#endif // OHOS_OES_TO_2D_RENDERER_H
