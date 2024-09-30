
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
#ifndef GPU_TENSOR_H_
#define GPU_TENSOR_H_

#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include <cuda_fp16.h>
namespace gpu {
template <typename T>
class gpu_tensor {
public:
    gpu_tensor();

    gpu_tensor(int32_t batch, int32_t channel, int32_t height, int32_t width);

    gpu_tensor(int32_t batch, int32_t channel, int32_t height, int32_t width, T *ptr);

    gpu_tensor(gpu_tensor<T> const &);

    gpu_tensor &operator=(const gpu_tensor<T> &);

    gpu_tensor(gpu_tensor<T> &&) noexcept ;

    gpu_tensor &operator=(gpu_tensor<T> &&) noexcept;

    ~gpu_tensor();

    int32_t get_width() const;

    int32_t get_height() const;

    int32_t get_channel() const;

    int32_t get_batch() const;

    /**
     * @brief re-alloc a gpu buffer to tensor with new size
     * @warning no data copy, original data will be lost
     * */
    void resize(int32_t batch, int32_t channel, int32_t height, int32_t width);

    void set_width(int32_t width);

    void set_height(int32_t height);

    void set_channel(int32_t channel);

    void set_batch(int32_t batch);

    /** TODO stride implement */

    T *get_buffer() const;

    const std::string &get_name();

    void set_name(const std::string &name);

    void set_value(T value, int64_t stream = 0);

    /** gpu tensor helper function */
    void write(const std::string &path) const;

    void load(const std::string &path);

    std::vector<T> to_vector() const;

    void from_vector(const std::vector<T> &vec);

private:
    class impl;

    std::unique_ptr<impl> ptr_;
};
}  // namespace gpu

#endif  // GPU_TENSOR_H
