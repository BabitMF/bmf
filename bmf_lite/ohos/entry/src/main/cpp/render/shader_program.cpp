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
#include "shader_program.h"

#include "common/common.h"
#include <hilog/log.h>

namespace bmf_lite_demo {
ShaderProgram::ShaderProgram(const std::string &vertexShader,
                             const std::string &fragShader) {
    auto vShaderCode = vertexShader.c_str();
    GLuint vertex = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertex, 1, &vShaderCode, nullptr);
    glCompileShader(vertex);
    CheckCompileErrors(vertex, "VERTEX");

    auto fShaderCode = fragShader.c_str();
    GLuint fragment = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragment, 1, &fShaderCode, nullptr);
    glCompileShader(fragment);
    CheckCompileErrors(fragment, "FRAGMENT");

    id_ = glCreateProgram();
    glAttachShader(id_, vertex);
    glAttachShader(id_, fragment);
    glLinkProgram(id_);
    CheckCompileErrors(id_, "PROGRAM");
    glDeleteShader(vertex);
    glDeleteShader(fragment);
}

ShaderProgram::~ShaderProgram() noexcept {
    if (Valid()) {
        glDeleteProgram(id_);
    }
}

void ShaderProgram::CheckCompileErrors(GLuint shader,
                                       const std::string &shaderType) {
    int success;
    char infoLog[1024];
    if (shaderType != "PROGRAM") {
        glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
        if (!success) {
            glGetShaderInfoLog(shader, 1024, NULL, infoLog);
            OH_LOG_Print(LOG_APP, LOG_ERROR, LOG_PRINT_DOMAIN, "ShaderProgram",
                         "ERROR::SHADER_COMPILATION_ERROR of type: %{public}s, "
                         "infoLog is: %{public}s",
                         shaderType.c_str(), infoLog);
        }
    } else {
        glGetProgramiv(shader, GL_LINK_STATUS, &success);
        if (!success) {
            glGetProgramInfoLog(shader, 1024, NULL, infoLog);
            OH_LOG_Print(LOG_APP, LOG_ERROR, LOG_PRINT_DOMAIN, "ShaderProgram",
                         "ERROR::PROGRAM_LINKING_ERROR of type: %{public}s, "
                         "infoLog is: %{public}s",
                         shaderType.c_str(), infoLog);
        }
    }
}

void ShaderProgram::SetBool(const std::string &name, bool value) {
    glUniform1i(glGetUniformLocation(id_, name.c_str()),
                static_cast<GLint>(value));
}

void ShaderProgram::SetInt(const std::string &name, int value) {
    glUniform1i(glGetUniformLocation(id_, name.c_str()),
                static_cast<GLint>(value));
}

void ShaderProgram::SetFloat(const std::string &name, float value) {
    glUniform1f(glGetUniformLocation(id_, name.c_str()),
                static_cast<GLfloat>(value));
}

void ShaderProgram::SetFloat4v(const std::string &name, float *values,
                               int cnt) {
    if (cnt != 4 || values == nullptr) {
        OH_LOG_Print(LOG_APP, LOG_ERROR, LOG_PRINT_DOMAIN, "ShaderProgram",
                     "ShaderProgram::SetFloat4v: invalid arguments.");
        return;
    }
    glUniform4fv(glGetUniformLocation(id_, name.c_str()), 1, values);
}

void ShaderProgram::SetMatrix4v(const std::string &name, float *matrix, int cnt,
                                bool transpose) {
    if (cnt != 16 || matrix == nullptr) {
        OH_LOG_Print(LOG_APP, LOG_ERROR, LOG_PRINT_DOMAIN, "ShaderProgram",
                     "ShaderProgram::SetFloat4v: invalid arguments.");
        return;
    }
    GLboolean glTranspose = transpose ? GL_TRUE : GL_FALSE;
    glUniformMatrix4fv(glGetUniformLocation(id_, name.c_str()), 1, glTranspose,
                       matrix);
}

GLint ShaderProgram::GetAttribLocation(const std::string &name) {
    return glGetAttribLocation(id_, name.c_str());
}
} // namespace bmf_lite_demo
