/*
 * Copyright 2023 Babit Authors
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
#ifndef _BMF_ALGORITHM_MODULES_OPS_OPENGL_UTILS_H_
#define _BMF_ALGORITHM_MODULES_OPS_OPENGL_UTILS_H_

#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <cstring>
#include <GLES3/gl31.h>
#include "utils/macros.h"
#include "utils/SHA256.h"

namespace bmf_lite {
namespace opengl {

class GLHelper {
  private:
    GLHelper() = default;

  public:
    static GLHelper &instance() {
        static GLHelper *g_utils_ins = nullptr;
        static std::once_flag s_once_flag;

        std::call_once(s_once_flag, [&]() { g_utils_ins = new GLHelper(); });
        return *g_utils_ins;
    }

    GLHelper(const GLHelper &) = delete;
    GLHelper &operator=(const GLHelper &) = delete;

  public:
    int build_program(GLuint *program_id, const std::string &cs,
                      const std::string &custom_head,
                      const std::string &program_cache_dir, GLint *local_size,
                      GLint local_x, GLint local_y, GLint local_z) {
        OPS_CHECK(local_size != nullptr, "local_size is nullptr");

        std::string header;
        OPS_CHECK(BMF_LITE_StsOk ==
                      get_local_size_and_header(local_size, header, custom_head,
                                                local_x, local_y, local_z),
                  "gen_local_size_and_header error");
        std::string complete_source = header + cs;

        SHA256 sha256;
        sha256.update(complete_source);
        std::string program_cache_name = SHA256::toString(sha256.digest());

        if (program_cache_dir.empty()) {
            OPS_CHECK(BMF_LITE_StsOk == build_program_with_source(
                                            complete_source, program_id),
                      "build_program error");
        } else {
            std::string program_cache_file_path =
                program_cache_dir + "/" + program_cache_name;
            if (!load_program(program_cache_file_path, program_id,
                              local_size)) {
                OPS_LOG_WARN("load_program from: %s fail",
                             program_cache_file_path.c_str());
                OPS_CHECK(BMF_LITE_StsOk == build_program_with_source(
                                                complete_source, program_id),
                          "load then build_program error");
                if (BMF_LITE_StsOk !=
                    save_program(program_cache_file_path, *program_id,
                                 local_size[0], local_size[1], local_size[2])) {
                    OPS_LOG_WARN("save_program to %s fail",
                                 program_cache_name.c_str());
                }
            }
        }
        return BMF_LITE_StsOk;
    }

    GLuint gen_itex(GLsizei width, GLsizei height, GLenum internalformat,
                    GLenum target, GLint min_mag_filter) {
        GLuint tex_id;
        glGenTextures(1, &tex_id);
        glBindTexture(target, tex_id);
        glTexParameteri(target, GL_TEXTURE_MIN_FILTER, min_mag_filter);
        glTexParameteri(target, GL_TEXTURE_MAG_FILTER, min_mag_filter);
        glTexParameteri(target, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(target, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexParameteri(target, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
        glTexStorage2D(target, 1, internalformat, width, height);
        return tex_id;
    }

    GLuint gen_tex(GLsizei width, GLsizei height, GLenum internalformat,
                   GLenum format, GLenum type, const void *data, GLenum target,
                   GLint min_mag_filter) {
        GLuint tex_id;
        glGenTextures(1, &tex_id);
        glBindTexture(target, tex_id);
        glTexParameteri(target, GL_TEXTURE_MIN_FILTER, min_mag_filter);
        glTexParameteri(target, GL_TEXTURE_MAG_FILTER, min_mag_filter);
        glTexParameteri(target, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(target, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexParameteri(target, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
        glTexImage2D(target, 0, internalformat, width, height, 0, format, type,
                     data);
        return tex_id;
    }

  private:
    int get_local_size_and_header(GLint *local_size, std::string &output_header,
                                  const std::string &custom_head, GLint local_x,
                                  GLint local_y, GLint local_z) {
        std::ostringstream shader_common_head;
        shader_common_head << "#version 310 es\n";
        if (!custom_head.empty()) {
            shader_common_head << custom_head << "\n";
        }

        GLint tmp_local_x{0}, tmp_local_y{0}, tmp_local_z{0};

        if (local_x == 0 && local_y != 0 && local_z != 0) {
            int x{256};
            while (x * local_y * local_z > max_total_local_size() ||
                   x > max_local_size_x()) {
                x /= 2;
            }
            tmp_local_x = x;
            tmp_local_y = local_y;
            tmp_local_z = local_z;
        } else if (local_x == 0 && local_y == 0 && local_z != 0) {
            int x{16}, y{16};
            while (x * y * local_z > max_total_local_size() ||
                   x > max_local_size_x() || y > max_local_size_y()) {
                if (x > y) {
                    x /= 2;
                } else {
                    y /= 2;
                }
            }

            tmp_local_x = x;
            tmp_local_y = y;
            tmp_local_z = local_z;
        } else if (local_x == 0 && local_y == 0 & local_z == 0) {
            int x{8}, y{8}, z{8};
            while (x * y * z > max_total_local_size() ||
                   x > max_local_size_x() || y > max_local_size_y() ||
                   z > max_local_size_z()) {
                if (x > y) {
                    x /= 2;
                } else if (y > z) {
                    y /= 2;
                } else {
                    z /= 2;
                }
            }

            tmp_local_x = x;
            tmp_local_y = y;
            tmp_local_z = z;
        } else if (local_x != 0 && local_y != 0 && local_z != 0) {
            tmp_local_x = local_x;
            tmp_local_y = local_y;
            tmp_local_z = local_z;
        } else {
            OPS_LOG_ERROR(
                "invalid local size: local_x:%d, local_y%d, local_z:%d",
                local_x, local_y, local_z);
            return BMF_LITE_OpsError;
        }

        OPS_CHECK(
            tmp_local_x * tmp_local_y * tmp_local_z <= max_total_local_size() &&
                tmp_local_x <= max_local_size_x() &&
                tmp_local_y <= max_local_size_y() &&
                tmp_local_z <= max_local_size_z() && tmp_local_x > 0 &&
                tmp_local_y > 0 && tmp_local_z > 0,
            "invalid local size with input local_x:%d, local_y:%d, local_z:%d",
            local_x, local_y, local_z);

        local_size[0] = tmp_local_x;
        local_size[1] = tmp_local_y;
        local_size[2] = tmp_local_z;

        shader_common_head << "layout (local_size_x = " << tmp_local_x << ",";
        shader_common_head << "local_size_y = " << tmp_local_y << ",";
        shader_common_head << "local_size_z =" << tmp_local_z << ") in;\n";

        output_header = shader_common_head.str();
        return BMF_LITE_StsOk;
    }

    int load_program(const std::string &program_cache_file_path,
                     GLuint *program_id, GLint *local_size) {
        std::ifstream ifile(program_cache_file_path);
        if (!ifile) {
            return BMF_LITE_OpsError;
        }
        std::stringstream buffer;
        buffer << ifile.rdbuf();
        ifile.close();

        std::string file_content_str = buffer.str();
        size_t bin_size = file_content_str.size();
        const auto local_xyz_size = 3 * sizeof(GLint);
        const auto binary_format_size = sizeof(GLenum);
        if (bin_size < local_xyz_size + binary_format_size) {
            OPS_LOG_ERROR("invalid bin_size: %zu", bin_size);
            return BMF_LITE_OpsError;
        }
        const char *file_content_char_ptr = file_content_str.c_str();

        memcpy(local_size, file_content_char_ptr, local_xyz_size);

        GLenum *binary_format_ptr =
            (GLenum *)(file_content_char_ptr + local_xyz_size);
        void *binary = (void *)(file_content_char_ptr + local_xyz_size +
                                binary_format_size);

        GLsizei length = bin_size - local_xyz_size - binary_format_size;
        *program_id = glCreateProgram();
        glProgramBinary(*program_id, *binary_format_ptr, binary, length);
        OPS_CHECK_OPENGL;

        GLint linked = 0;
        glGetProgramiv(*program_id, GL_LINK_STATUS, &linked);
        if (!linked) {
            OPS_LOG_ERROR("link error");
            return BMF_LITE_OpsError;
        }
        return BMF_LITE_StsOk;
    }

    int save_program(const std::string &program_cache_file_path,
                     GLuint program_id, GLint local_x, GLint local_y,
                     GLint local_z) {
        std::ofstream ofile(program_cache_file_path);
        if (!ofile) {
            return BMF_LITE_OpsError;
        }

        GLenum binary_foramt = 0;
        // GLsizei length;
        GLint buf_size = 0;
        void *binary = nullptr;

        glGetProgramiv(program_id, GL_PROGRAM_BINARY_LENGTH, &buf_size);
        if (buf_size <= 0) {
            return BMF_LITE_OpsError;
        }
        std::vector<uint8_t> bianry_vec(buf_size);
        binary = bianry_vec.data();

        glGetProgramBinary(program_id, buf_size, NULL, &binary_foramt, binary);
        OPS_CHECK_OPENGL;

        // OPS_LOG_ERROR("binary size: %d binary_foramt: %d", buf_size,
        // binary_foramt);

        std::vector<char> binary_chars(sizeof(GLint) * 3 + sizeof(GLenum) +
                                       buf_size);
        std::memcpy(binary_chars.data(), &local_x, sizeof(GLint));
        std::memcpy(binary_chars.data() + sizeof(GLint), &local_y,
                    sizeof(GLint));
        std::memcpy(binary_chars.data() + sizeof(GLint) * 2, &local_z,
                    sizeof(GLint));
        std::memcpy(binary_chars.data() + sizeof(GLint) * 3, &binary_foramt,
                    sizeof(GLenum));
        std::memcpy(binary_chars.data() + sizeof(GLint) * 3 + sizeof(GLenum),
                    binary, buf_size);

        std::string binary_str(binary_chars.begin(), binary_chars.end());
        ofile << binary_str;
        ofile.close();
        return BMF_LITE_StsOk;
    }

    int build_program_with_source(const std::string &source,
                                  GLuint *program_id) {
        GLuint shader_id = glCreateShader(GL_COMPUTE_SHADER);

        auto source_c_str = source.c_str();
        glShaderSource(shader_id, 1, &source_c_str, NULL);
        glCompileShader(shader_id);

        GLint status;
        glGetShaderiv(shader_id, GL_COMPILE_STATUS, &status);
        if (!status) {
            int len = 0;
            glGetShaderiv(shader_id, GL_INFO_LOG_LENGTH, &len);
            if (0 >= len) {
                glGetShaderInfoLog(shader_id, 0, &len, NULL);
            }
            char *buffer = new char[len + 1];
            glGetShaderInfoLog(shader_id, len, NULL, buffer);
            buffer[len] = 0;
            OPS_LOG_ERROR("compile log: %s", buffer);
            delete[] buffer;
            glDeleteShader(shader_id);
            return BMF_LITE_OpsError;
        }
        *program_id = glCreateProgram();

        glAttachShader(*program_id, shader_id);
        glLinkProgram(*program_id);

        GLint linked;
        glGetProgramiv(*program_id, GL_LINK_STATUS, &linked);
        if (!linked) {
            GLsizei len;
            glGetProgramiv(*program_id, GL_INFO_LOG_LENGTH, &len);
            if (len <= 0) {
                glGetProgramInfoLog(*program_id, 0, &len, NULL);
            }
            if (len > 0) {
                char *buffer = new char[len + 1];
                buffer[len] = '\0';
                glGetProgramInfoLog(*program_id, len, NULL, buffer);
                OPS_LOG_ERROR("link log: %s", buffer);
                delete[] buffer;
                glDeleteShader(shader_id);
                return BMF_LITE_OpsError;
            }
        }
        glDeleteShader(shader_id);
        OPS_CHECK_OPENGL;
        return BMF_LITE_StsOk;
    }

    GLint max_local_size_x() {
        if (max_local_size_x_ == 0) {
            glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_SIZE, 0,
                            &max_local_size_x_);
        }
        return max_local_size_x_;
    }
    GLint max_local_size_y() {
        if (max_local_size_y_ == 0) {
            glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_SIZE, 1,
                            &max_local_size_y_);
        }
        return max_local_size_y_;
    }
    GLint max_local_size_z() {
        if (max_local_size_z_ == 0) {
            glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_SIZE, 2,
                            &max_local_size_z_);
        }
        return max_local_size_z_;
    }
    GLint max_total_local_size() {
        if (max_total_local_size_ == 0) {
            glGetIntegerv(GL_MAX_COMPUTE_WORK_GROUP_INVOCATIONS,
                          &max_total_local_size_);
        }
        return max_total_local_size_;
    }

  private:
    GLint max_local_size_x_ = 0;
    GLint max_local_size_y_ = 0;
    GLint max_local_size_z_ = 0;
    GLint max_total_local_size_ = 0;
}; // class utils
} // namespace opengl
} // namespace bmf_lite

#endif