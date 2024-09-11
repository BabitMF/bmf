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
#ifndef NATIVEIMAGEDEMO_SHADER_PROGRAM_H
#define NATIVEIMAGEDEMO_SHADER_PROGRAM_H

#include <string>

#include <GLES3/gl3.h>

namespace bmf_lite_demo {
class ShaderProgram {
  public:
    ShaderProgram(const std::string &vertexShader,
                  const std::string &fragShader);
    ~ShaderProgram() noexcept;

    // disallow copy and move
    ShaderProgram(const ShaderProgram &other) = delete;
    void operator=(const ShaderProgram &other) = delete;
    ShaderProgram(ShaderProgram &&other) = delete;
    void operator=(ShaderProgram &&other) = delete;

    bool Valid() const { return id_ > 0; }
    void Use() const { glUseProgram(id_); }

    void SetBool(const std::string &name, bool value);
    void SetInt(const std::string &name, int value);
    void SetFloat(const std::string &name, float value);
    void SetFloat4v(const std::string &name, float *values, int cnt);
    void SetMatrix4v(const std::string &name, float *matrix, int cnt,
                     bool transpose = false);
    GLint GetAttribLocation(const std::string &name);

  private:
    void CheckCompileErrors(GLuint shader, const std::string &shaderType);
    GLuint id_ = 0;
};
} // namespace bmf_lite_demo
#endif // NATIVEIMAGEDEMO_SHADER_PROGRAM_H
