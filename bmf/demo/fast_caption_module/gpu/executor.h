/*
    Copyright 2024 Babit Authors
    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
*/
#ifndef EXECUTOR_H_
#define EXECUTOR_H_

#include "gpu_tensor.h"
#include <cuda_fp16.h>
#include <functional>
#include <unordered_map>

namespace gpu {
/** executor logger setup */
using log_callback_t = std::function<void(int, const std::string &)>;
void set_logger(log_callback_t cb);

class executor /* no copy */
{
public:
    executor();

    executor(const executor &) = delete;

    executor &operator=(const executor &) = delete;

public:
    explicit executor(const std::string &path);

    executor(const uint8_t *data, size_t length);

    ~executor();

    /**
     * @brief load model
     * @param alloc using internal memory if true, otherwise using external memory
     * */
    bool load(const std::string &path, bool alloc = true);
    bool load(const uint8_t *data, size_t length, bool alloc = true);

    /**
     * @brief set external memory
     */
    void set_device_memory(void *ptr);
    size_t get_device_memory();

    /** get model io info */
    int get_model_num();
    int get_input_num(int32_t index = 0);
    int get_output_num(int32_t index = 0);
    std::string get_binding_name(int32_t model_idx, int32_t binding_idx);

    /** @return return two dimensions about binding. index-0 is min dims, while index-1 is max dim. */
    std::array<std::vector<int32_t>, 2>
    get_binding_dim(int32_t model_idx, const std::string &binding_name);

    /** get model detailed output info */
    std::unordered_map<std::string, std::vector<int32_t>>
    get_output_shapes(std::unordered_map<std::string, std::vector<int32_t>> input_info);

public:
    /** model inference */
    int execute_dynamic(const std::vector<const gpu_tensor<float> *> &bindings, int64_t stream = 0);
    int execute_dynamic(const std::vector<const gpu_tensor<half> *> &bindings, int64_t stream = 0);

    int execute_dynamic(std::unordered_map<std::string, const gpu_tensor<float> *> bindings, int64_t stream = 0);
    int execute_dynamic(std::unordered_map<std::string, const gpu_tensor<half> *> bindings, int64_t stream = 0);

private:
    /** hidden implement */
    class impl;

    std::unique_ptr<impl> ptr_;
};
}// namespace gpu
#endif// EXECUTOR_H_
