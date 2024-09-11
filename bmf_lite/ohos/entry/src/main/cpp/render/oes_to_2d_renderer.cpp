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
#include "oes_to_2d_renderer.h"

#include <string>
#include <GLES2/gl2ext.h>

namespace bmf_lite_demo {

namespace oes_shader {
std::string vertexShader = R"delimiter(
attribute vec3 position;
attribute vec2 texCoord;

varying vec2 vTexCoord;

uniform mat4 matTransform;

void main()
{
    gl_Position = matTransform * vec4(position, 1.0);
    vTexCoord = texCoord;
}
)delimiter";

std::string fragmentShader = R"delimiter(
#extension GL_OES_EGL_image_external : require
precision highp float;
varying vec2 vTexCoord;
uniform samplerExternalOES texture;

void main()
{
    gl_FragColor = texture2D(texture, vTexCoord).rgba;
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

} // namespace oes_shader

OesTo2dRenderer::OesTo2dRenderer() {
    textureOesShaderProgram_ = std::make_unique<ShaderProgram>(
        oes_shader::vertexShader, oes_shader::fragmentShader);
}

OesTo2dRenderer::~OesTo2dRenderer() {
    glDeleteVertexArrays(1, &vertexArrayObject_);
    vertexArrayObject_ = GL_NONE;
    glDeleteBuffers(1, &vertexBufferObject_);
    vertexBufferObject_ = GL_NONE;
    textureOesShaderProgram_.reset();
    if (frameBuffer_ != GL_NONE) {
        glDeleteFramebuffers(1, &frameBuffer_);
        frameBuffer_ = GL_NONE;
    }
}

int OesTo2dRenderer::init() {
    if (!textureOesShaderProgram_->Valid()) {
        return -1;
    }
    glGenVertexArrays(1, &vertexArrayObject_);
    glGenBuffers(1, &vertexBufferObject_);

    glBindVertexArray(vertexArrayObject_);
    glBindBuffer(GL_ARRAY_BUFFER, vertexBufferObject_);
    glBufferData(GL_ARRAY_BUFFER, sizeof(oes_shader::vertices),
                 oes_shader::vertices, GL_STATIC_DRAW);
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

void OesTo2dRenderer::setMatrix(std::vector<float> matrix) {
    matrix_ = std::move(matrix);
}

int OesTo2dRenderer::process(GLuint input_texture, GLuint output_texture,
                             int width, int height) {
    glViewport(0, 0, width, height);
    GLint old_fbo = 0;
    glGetIntegerv(GL_FRAMEBUFFER_BINDING, &old_fbo);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_EXTERNAL_OES, input_texture);
    textureOesShaderProgram_->Use();
    textureOesShaderProgram_->SetInt("texture", 0);
    glBindTexture(GL_TEXTURE_2D, output_texture);
    glBindFramebuffer(GL_FRAMEBUFFER, frameBuffer_);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D,
                           output_texture, 0);
    int val = glCheckFramebufferStatus(GL_FRAMEBUFFER);
    if (val != GL_FRAMEBUFFER_COMPLETE) {
        return -1;
    }
    textureOesShaderProgram_->SetMatrix4v("matTransform", matrix_.data(), 16,
                                          false);
    glBindVertexArray(vertexArrayObject_);
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, oes_shader::indices);
    glBindVertexArray(0);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D,
                           0, 0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    return 0;
}

int OesTo2dRenderer::drawToScreen(GLuint input_texture, int width, int height) {
    glViewport(0, 0, width, height);
    textureOesShaderProgram_->Use();
    textureOesShaderProgram_->SetInt("texture", 0);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_EXTERNAL_OES, input_texture);
    textureOesShaderProgram_->SetMatrix4v("matTransform", matrix_.data(), 16,
                                          false);
    glBindVertexArray(vertexArrayObject_);
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, oes_shader::indices);
    glBindVertexArray(0);
    return 0;
}
} // namespace bmf_lite_demo