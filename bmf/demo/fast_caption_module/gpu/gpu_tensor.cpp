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
#include "gpu_tensor.h"
#include <cuda_runtime_api.h>
#include <glog/logging.h>

#include <fstream>
#include <iomanip>
#include <typeinfo>
#include <unordered_map>

namespace gpu {

/**
 * gpu tensor implement
 */
template<typename T>
class gpu_tensor<T>::impl {
public:
    impl() = default;

    ~impl() {
        if (buffer_ && !assign_ptr_) {
            auto status = cudaFree(buffer_);
            if (status != cudaSuccess) {
                LOG(ERROR) << "free buffer error. " << cudaGetErrorName(status) << ": " << cudaGetErrorString(status);
            }
            VLOG_IF(3, status == cudaSuccess) << "free buffer success. buffer size is: " << real_size_;
            buffer_ = nullptr;
        }
    }

    int get_width() const { return width_; }

    void set_width(int width) {
        width_ = width;
        if (assign_ptr_) {
            assign_w_ = width;
        }
        alloc();
    }

    int get_height() const { return height_; }

    void set_height(int height) {
        height_ = height;
        if (assign_ptr_) {
            assign_h_ = height;
        }
        alloc();
    }

    int get_channel() const { return channel_; }

    void set_channel(int channel) {
        channel_ = channel;
        if (assign_ptr_) {
            assign_c_ = channel;
        }
        alloc();
    }

    int get_batch() const { return batch_; }

    void set_batch(int batch) {
        batch_ = batch;
        if (assign_ptr_) {
            assign_n_ = batch;
        }
        alloc();
    }

    size_t get_count() const {
        return size_t(batch_) * channel_ * height_ * width_;
    }

    T *get_buffer() const {
        if (!assign_ptr_) {
            if (get_count() != 0 && buffer_ == nullptr) {
                throw std::runtime_error("get_buffer on invalid gpu_tensor");
            }
        }

        return buffer_;
    }

    const std::string &get_name() const { return name_; }

    void set_name(const std::string &name) { name_ = name; }

    void set_value(int8_t value, int64_t stream = 0) {
        cudaMemsetAsync(buffer_, value, get_count() * sizeof(T), (cudaStream_t) stream);
    }

    void write(const std::string &path) {
        std::vector<T> host(get_count(), 0);
        cudaMemcpy(host.data(), buffer_, host.size() * sizeof(T), cudaMemcpyDeviceToHost);
        std::fstream file(path, std::ios::binary | std::ios::out);
        if (file.is_open()) {
            file.write((const char *) host.data(), host.size() * sizeof(T));
            file.close();
        }
    }

    void load(const std::string &path) {
        std::vector<T> host(get_count(), 0);
        std::fstream file(path, std::ios::binary | std::ios::in);
        if (file.is_open()) {
            auto rd_buf = file.rdbuf();
            size_t size = rd_buf->pubseekoff(std::ios::beg, std::ios::end, std::ios::in);
            LOG_IF(WARNING, size != host.size() * sizeof(T))
                    << "file size(" << size << ") does not match tensor size(" << host.size() << ")";
            rd_buf->pubseekpos(std::ios::beg, std::ios::in);
            rd_buf->sgetn((char *) host.data(), std::min(size, host.size() * sizeof(T)));
            cudaMemcpy(buffer_, host.data(), std::min(size, host.size() * sizeof(T)), cudaMemcpyHostToDevice);
            file.close();
        } else {
            LOG(ERROR) << "file does not exists: " << path;
        }
    }

    std::vector<T> to_vector() const {
        std::vector<T> host(get_count(), 0);
        cudaMemcpy(host.data(), buffer_, sizeof(T) * host.size(), cudaMemcpyDeviceToHost);
        return host;
    }

    void from_vector(const std::vector<T> &vec) {
        size_t size = get_count();
        LOG_IF(WARNING, vec.size() != size) << "vector size(" << vec.size() << ") does not match tensor size(" << size << ")";
        cudaMemcpy(buffer_, vec.data(), std::min(size, vec.size()) * sizeof(T), cudaMemcpyHostToDevice);
    }

    void resize(int32_t batch, int32_t channel, int32_t height, int32_t width) {
        if (!assign_ptr_) {
            if (batch_ != batch || channel_ != channel || height_ != height || width_ != width) {
                batch_ = batch;
                channel_ = channel;
                height_ = height;
                width_ = width;

                size_t new_size = get_count() * sizeof(T);

                if (new_size > real_size_) {
                    /** whether copy origin data? temporal no*/
                    auto status = cudaFree(buffer_);
                    if (status != cudaSuccess) {
                        LOG(ERROR) << "resize error. free old buffer error. " << cudaGetErrorName(status) << ": " << cudaGetErrorString(status);
                    }
                    buffer_ = nullptr;
                    status = cudaMalloc((void **) &buffer_, new_size);
                    if (status != cudaSuccess) {
                        LOG(ERROR) << "resize error. malloc new buffer error. " << cudaGetErrorName(status) << ": " << cudaGetErrorString(status);
                    }
                    real_size_ = new_size;
                }
            }
        } else {
            LOG_IF(WARNING, batch > assign_n_) << "exceed assigned batch size " << assign_n_;
            LOG_IF(WARNING, channel > assign_c_) << "exceed assigned channel size " << assign_c_;
            LOG_IF(WARNING, height > assign_h_) << "exceed assigned height size " << assign_h_;
            LOG_IF(WARNING, width > assign_w_) << "exceed assigned width size " << assign_w_;
            batch_ = batch;
            channel_ = channel;
            height_ = height;
            width_ = width;
        }
    }

    void set_assign_ptr(T *ptr) {
        if (buffer_ && !assign_ptr_) {
            LOG(WARNING) << "buffer has alloced, this action will free pre buffer";
            cudaFree(buffer_);
        }
        if (ptr == nullptr) {
            LOG(WARNING) << "assigned null pointer";
        }
        buffer_ = ptr;
        assign_ptr_ = true;
    }

private:
    inline bool alloc() {
        size_t size = get_count() * sizeof(T);
        if (size && !buffer_) {
            VLOG(3) << "apply buffer length: " << size << ". element size: " << sizeof(T) << ". width: " << width_
                    << ". height: " << height_ << ". channel: " << channel_ << ". batch: " << batch_;
            assign_ptr_ = false;
            real_size_ = size;
            auto status = cudaMalloc((void **) &buffer_, size);

            if (status != cudaSuccess) {
                size_t free, total;
                cudaMemGetInfo(&free, &total);
                size_t allocated = total - free;

                LOG(ERROR) << std::fixed << std::setprecision(2)
                           << cudaGetErrorName(status) << " : " << cudaGetErrorString(status)
                           << ". buffer is " << (void *) buffer_
                           << ". Tried to allocate " << size / 1048576.f
                           << " MB. " << total / 1073741824.f << " GB total capacity; "
                           << allocated / 1073741824.f << " GB already allocated; "
                           << free / 1073741824.f << " GB free;";
                buffer_ = nullptr;
                return false;
            }
            VLOG(3) << "apply buffer is: " << (void *) buffer_;
            return cudaSuccess == status;
        } else {
            VLOG_IF(3, size != 0) << "alloc failed. size " << size << "="
                                  << batch_ << "x" << channel_ << "x" << height_ << "x" << width_
                                  << " or buffer " << (void *) buffer_ << " is already alloc.";
            return false;
        }
    }

    int width_ = 0;
    int height_ = 0;
    int channel_ = 0;
    int batch_ = 0;
    T *buffer_ = nullptr;
    size_t real_size_ = 0;

    std::string name_ = "";
    bool assign_ptr_ = false;
    int assign_w_ = 0;
    int assign_h_ = 0;
    int assign_c_ = 0;
    int assign_n_ = 0;
};

template<typename T>
gpu_tensor<T>::gpu_tensor() {
    ptr_ = std::unique_ptr<impl>(new impl);
}

template<typename T>
gpu_tensor<T>::gpu_tensor(int batch, int channel, int height, int width) : gpu_tensor() {
    ptr_->set_batch(batch);
    ptr_->set_channel(channel);
    ptr_->set_height(height);
    ptr_->set_width(width);
}

template<typename T>
gpu_tensor<T>::gpu_tensor(int32_t batch, int32_t channel, int32_t height, int32_t width, T *ptr) : gpu_tensor() {
    ptr_->set_assign_ptr(ptr);
    ptr_->set_batch(batch);
    ptr_->set_channel(channel);
    ptr_->set_height(height);
    ptr_->set_width(width);


    if (ptr_->get_count() != 0 && ptr == nullptr) {
        throw std::runtime_error("null ptr with non-zero size");
    }
}

template<typename T>
gpu_tensor<T>::gpu_tensor(gpu_tensor<T> const &origin) : gpu_tensor() {
    ptr_->set_batch(origin.get_batch());
    ptr_->set_channel(origin.get_channel());
    ptr_->set_height(origin.get_height());
    ptr_->set_width(origin.get_width());
    if (get_buffer() && origin.get_buffer()) {
        cudaMemcpy(ptr_->get_buffer(),
                   origin.get_buffer(),
                   sizeof(T) * origin.get_width() * origin.get_height() * origin.get_channel() * origin.get_batch(),
                   cudaMemcpyDeviceToDevice);
    }
}

template<typename T>
gpu_tensor<T> &gpu_tensor<T>::operator=(const gpu_tensor<T> &origin) {
    this->set_batch(origin.get_batch());
    this->set_channel(origin.get_channel());
    this->set_height(origin.get_height());
    this->set_width(origin.get_width());
    if (this->get_buffer() && origin.get_buffer()) {
        cudaMemcpy(this->get_buffer(),
                   origin.get_buffer(),
                   sizeof(T) * origin.get_width() * origin.get_height() * origin.get_channel() * origin.get_batch(),
                   cudaMemcpyDeviceToDevice);
    }
    return *this;
}

template<typename T>
gpu_tensor<T>::gpu_tensor(gpu_tensor<T> &&origin) noexcept : gpu_tensor() {
    ptr_.swap(origin.ptr_);
}

template<typename T>
gpu_tensor<T> &gpu_tensor<T>::operator=(gpu_tensor<T> &&origin) noexcept {
    ptr_.swap(origin.ptr_);
    return *this;
}

template<typename T>
int32_t gpu_tensor<T>::get_width() const {
    return ptr_->get_width();
}

template<typename T>
int32_t gpu_tensor<T>::get_height() const {
    return ptr_->get_height();
}

template<typename T>
int32_t gpu_tensor<T>::get_channel() const {
    return ptr_->get_channel();
}

template<typename T>
int32_t gpu_tensor<T>::get_batch() const {
    return ptr_->get_batch();
}

template<typename T>
void gpu_tensor<T>::resize(int32_t batch, int32_t channel, int32_t height, int32_t width) {
    ptr_->resize(batch, channel, height, width);
}

template<typename T>
void gpu_tensor<T>::set_width(int32_t width) {
    ptr_->set_width(width);
}

template<typename T>
void gpu_tensor<T>::set_height(int32_t height) {
    ptr_->set_height(height);
}

template<typename T>
void gpu_tensor<T>::set_channel(int32_t channel) {
    ptr_->set_channel(channel);
}

template<typename T>
void gpu_tensor<T>::set_batch(int32_t batch) {
    ptr_->set_batch(batch);
}

template<typename T>
T *gpu_tensor<T>::get_buffer() const {
    return ptr_->get_buffer();
}

template<typename T>
const std::string &gpu_tensor<T>::get_name() {
    return ptr_->get_name();
}

template<typename T>
void gpu_tensor<T>::set_name(const std::string &name) {
    ptr_->set_name(name);
}

template<typename T>
void gpu_tensor<T>::set_value(T value, int64_t stream) {
    ptr_->set_value(value);
}

template<typename T>
void gpu_tensor<T>::write(const std::string &path) const {
    ptr_->write(path);
}

template<typename T>
void gpu_tensor<T>::load(const std::string &path) {
    ptr_->load(path);
}

template<typename T>
std::vector<T> gpu_tensor<T>::to_vector() const {
    return ptr_->to_vector();
}

template<typename T>
void gpu_tensor<T>::from_vector(const std::vector<T> &vec) {
    ptr_->from_vector(vec);
}

template<typename T>
gpu_tensor<T>::~gpu_tensor() = default;

template class gpu_tensor<half>;

template class gpu_tensor<float>;

template class gpu_tensor<double>;

template class gpu_tensor<int8_t>;

template class gpu_tensor<uint8_t>;

template class gpu_tensor<int16_t>;

template class gpu_tensor<uint16_t>;
}// namespace gpu