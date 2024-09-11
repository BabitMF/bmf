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
#include "split_screen_renderer.h"
#include <string>
#include <GLES2/gl2ext.h>

namespace bmf_lite_demo {

namespace split_shader {
std::string vertexShader = R"delimiter(
uniform mat4 matTransform;
attribute vec3 position;
attribute vec2 texCoord;
varying vec2 vTexCoord;

void main()
{
    gl_Position = matTransform * vec4(position, 1.0);
    vTexCoord = texCoord;
}
)delimiter";

std::string fragmentShader = R"delimiter(
precision highp float;
uniform sampler2D leftTexture;
uniform sampler2D rightTexture;
uniform float splitRatio;
uniform int displayDivider;
varying vec2 vTexCoord;

void main()
{
    float divWidth = 0.005;
    if (displayDivider <= 0) {
        divWidth = 0.0;
    }
    if (vTexCoord.x <= splitRatio - divWidth) {
        gl_FragColor = texture2D(leftTexture, vTexCoord);
    } else if (vTexCoord.x > splitRatio - divWidth && vTexCoord.x <= splitRatio + divWidth && displayDivider > 0) {
        gl_FragColor = vec4(1.0, 0.5, 0.3, 1.0);
    } else {
        gl_FragColor = texture2D(rightTexture, vTexCoord);
    }
}
)delimiter";

GLfloat vertices[] = {
    // positions       // texture coords
    -1.0f, 1.0f,  0.0f, 0.0f, 1.0f, // top left
    -1.0f, -1.0f, 0.0f, 0.0f, 0.0f, // bottom left
    1.0f,  -1.0f, 0.0f, 1.0f, 0.0f, // bottom right
    1.0f,  1.0f,  0.0f, 1.0f, 1.0f  // top right
};
GLuint indices[] = {
    0, 1, 2, // first triangle
    0, 2, 3  // second triangle
};

} // namespace split_shader

SplitScreenRenderer::SplitScreenRenderer() {
    textureSplitShaderProgram_ = std::make_unique<ShaderProgram>(
        split_shader::vertexShader, split_shader::fragmentShader);
}

SplitScreenRenderer::~SplitScreenRenderer() {
    glDeleteVertexArrays(1, &vertexArrayObject_);
    vertexArrayObject_ = GL_NONE;
    glDeleteBuffers(1, &vertexBufferObject_);
    vertexBufferObject_ = GL_NONE;
    textureSplitShaderProgram_.reset();
    if (frameBuffer_ != GL_NONE) {
        glDeleteFramebuffers(1, &frameBuffer_);
        frameBuffer_ = GL_NONE;
    }
}

int SplitScreenRenderer::init() {
    if (!textureSplitShaderProgram_->Valid()) {
        return -1;
    }
    glGenVertexArrays(1, &vertexArrayObject_);
    glGenBuffers(1, &vertexBufferObject_);

    glBindVertexArray(vertexArrayObject_);
    glBindBuffer(GL_ARRAY_BUFFER, vertexBufferObject_);
    glBufferData(GL_ARRAY_BUFFER, sizeof(split_shader::vertices),
                 split_shader::vertices, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float),
                          (void *)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float),
                          (void *)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);

    glGenFramebuffers(1, &frameBuffer_);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
    return 0;
}

void SplitScreenRenderer::setDisplayDivider(int display_divider) {
    display_divider_ = display_divider;
}

void SplitScreenRenderer::setSplitRatio(float ratio) { split_ratio_ = ratio; }

void SplitScreenRenderer::setMatrix(std::vector<float> matrix) {
    matrix_ = std::move(matrix);
}

int SplitScreenRenderer::drawToScreen(GLuint left_texture, GLuint right_texture,
                                      int width, int height) {
    glViewport(0, 0, width, height);
    textureSplitShaderProgram_->Use();
    glBindVertexArray(vertexArrayObject_);
    textureSplitShaderProgram_->Use();
    textureSplitShaderProgram_->SetInt("leftTexture", 0);
    textureSplitShaderProgram_->SetInt("rightTexture", 1);
    textureSplitShaderProgram_->SetFloat("splitRatio", split_ratio_);
    textureSplitShaderProgram_->SetInt("displayDivider", display_divider_);
    textureSplitShaderProgram_->SetMatrix4v("matTransform", matrix_.data(), 16,
                                            false);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, left_texture);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, right_texture);
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, split_shader::indices);
    glBindVertexArray(0);
    return 0;
}

} // namespace bmf_lite_demo