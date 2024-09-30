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
#include <NvInfer.h>
#include "executor.h"
#include <cuda_runtime_api.h>
#include <cuda_fp16.h>
#include <glog/logging.h>

#include <fstream>
#include <memory>
#include <vector>
#include <cstdint>

namespace gpu {
/** logger setup */
class _logger : public nvinfer1::ILogger {
public:
    void log(Severity severity, nvinfer1::AsciiChar const *msg) noexcept override {
        if (log_func_) {
            log_func_(static_cast<int>(severity), std::string(msg));
        }
    }

    void set_log(log_callback_t cb) { log_func_ = cb; }

private:
    log_callback_t log_func_{};
};

static _logger trt_logger;

void set_logger(log_callback_t cb) { trt_logger.set_log(std::move(cb)); }

class nv_model {
public:
    nv_model() { runtime_ = nvinfer1::createInferRuntime(trt_logger); }

    nv_model(const uint8_t *data, size_t length) : nv_model() { load(data, length); }

    ~nv_model() {
        if (context_) {
            delete context_;
        }
        if (engine_) {
            delete engine_;
        }
        if (runtime_) {
            delete runtime_;
        }
    }

    bool load(const uint8_t *data, size_t length) {
        engine_ = runtime_->deserializeCudaEngine(data, length);
        if (engine_) {
            auto bindings_per_profile = engine_->getNbIOTensors() / engine_->getNbOptimizationProfiles();
            for (int n = 0; n < engine_->getNbOptimizationProfiles(); n++) {
                for (int i = 0; i < bindings_per_profile; i++) {
                    auto binding_index = i + n * bindings_per_profile;
                    auto name = engine_->getIOTensorName(binding_index);
                    m_tensor_names.emplace_back(name);
                    if (is_input(name)) {
                        auto min = engine_->getProfileShape(name, n, nvinfer1::OptProfileSelector::kMIN);
                        auto max = engine_->getProfileShape(name, n, nvinfer1::OptProfileSelector::kMAX);
                        VLOG(2) << "binding index :" << binding_index
                                << ". name is: " << name
                                << ". input size : ["
                                << min.d[0] << "-" << max.d[0] << "]x["
                                << min.d[1] << "-" << max.d[1] << "]x["
                                << min.d[2] << "-" << max.d[2] << "]x["
                                << min.d[3] << "-" << max.d[3] << "]";
                    } else {
                        VLOG(2) << "binding index :" << binding_index
                                << ". name is: " << name;
                    }
                }
            }
            context_ = engine_->createExecutionContextWithoutDeviceMemory();
            // bindings_.resize(engine_->getNbIOTensors(), nullptr);
            m_size = engine_->getDeviceMemorySize();
            return true;
        } else {
            return false;
        }
    }

    template<typename T>
    bool is_fit(int32_t index, const gpu_tensor<T> *tensor) {
        int32_t width = tensor->get_width();
        int32_t height = tensor->get_height();
        auto num_profiles = engine_->getNbOptimizationProfiles();
        auto bindings_per_profile = engine_->getNbIOTensors() / num_profiles;
        for (int i = 0; i < num_profiles; i++) {
            auto binding_index = index + i * bindings_per_profile;
            auto name = m_tensor_names[binding_index];
            if (is_input(name)) {
                auto min = engine_->getProfileShape(name, i, nvinfer1::OptProfileSelector::kMIN);
                auto max = engine_->getProfileShape(name, i, nvinfer1::OptProfileSelector::kMAX);
                if ((width >= min.d[min.nbDims - 1] && width <= max.d[max.nbDims - 1]) &&
                    (height >= min.d[min.nbDims - 2] && height <= max.d[max.nbDims - 2])) {
                    if (i != context_->getOptimizationProfile()) {
                        context_->setOptimizationProfileAsync(i, 0);
                    }
                    nvinfer1::Dims dims{};
                    if (min.nbDims == 4) {
                        dims = nvinfer1::Dims4(tensor->get_batch(), tensor->get_channel(), height, width);
                    } else {
                        dims.nbDims = min.nbDims;
                        for (int n = 0; n < dims.nbDims - 2; n++) {
                            dims.d[n] = min.d[n];
                        }
                        dims.d[dims.nbDims - 1] = width;
                        dims.d[dims.nbDims - 2] = height;
                    }
                    context_->setInputShape(name, dims);
                    bindings_[name] = tensor->get_buffer();
                    return true;
                }
            } else {
                VLOG(3) << "fit test. binding per profile:" << bindings_per_profile << ". num of opt profile:" << num_profiles
                        << ". binding_index:" << binding_index << ". profile_index:" << i << ". index:" << index;
                bindings_[name] = tensor->get_buffer();
                return true;
            }
        }
        return false;
    }

    template<typename T>
    bool is_fit(const std::string &name, const gpu_tensor<T> *tensor) {
        int32_t width = tensor->get_width();
        int32_t height = tensor->get_height();
        auto num_profiles = engine_->getNbOptimizationProfiles();
        auto bindings_per_profile = engine_->getNbIOTensors() / num_profiles;
        for (int i = 0; i < num_profiles; i++) {
            // auto binding_index = engine_->getBindingIndex(name.c_str()) + i * bindings_per_profile;
            if (is_input(name.c_str())) {
                auto min = engine_->getProfileShape(name.c_str(), i, nvinfer1::OptProfileSelector::kMIN);
                auto max = engine_->getProfileShape(name.c_str(), i, nvinfer1::OptProfileSelector::kMAX);
                if ((width >= min.d[min.nbDims - 1] && width <= max.d[max.nbDims - 1]) &&
                    (height >= min.d[min.nbDims - 2] && height <= max.d[max.nbDims - 2])) {
                    if (i != context_->getOptimizationProfile()) {
                        context_->setOptimizationProfileAsync(i, 0);
                    }
                    nvinfer1::Dims dims{};
                    if (min.nbDims == 4) {
                        dims = nvinfer1::Dims4(tensor->get_batch(), tensor->get_channel(), height, width);
                    } else {
                        dims.nbDims = min.nbDims;
                        for (int n = 0; n < dims.nbDims - 2; n++) {
                            dims.d[n] = min.d[n];
                        }
                        dims.d[dims.nbDims - 1] = width;
                        dims.d[dims.nbDims - 2] = height;
                    }
                    context_->setInputShape(name.c_str(), dims);
                    bindings_[name] = tensor->get_buffer();
                    return true;
                }
            } else {
                VLOG(3) << "fit test. binding per profile:" << bindings_per_profile << ". num of opt profile:" << num_profiles
                        << ". binding name:" << name << ". profile_index: " << i;
                bindings_[name] = tensor->get_buffer();
                return true;
            }
        }
        return false;
    }

    template<typename T>
    bool enqueue(const std::vector<const gpu_tensor<T> *> &bindings,
                 cudaStream_t stream, cudaEvent_t *event) {
        size_t bindings_size = bindings.size();
        for (size_t i = 0; i < bindings_size; i++) {
            context_->setTensorAddress(m_tensor_names[i], bindings[i]->get_buffer());
        }
        auto result = context_->enqueueV3(stream);
        for (auto &binding : bindings_) {
            binding.second = nullptr;
        }
        return result;
    }

    template<typename T>
    bool enqueue(std::unordered_map<std::string, const gpu_tensor<T> *> bindings,
                 cudaStream_t stream, cudaEvent_t *event) {
        auto num_profiles = engine_->getNbOptimizationProfiles();
        auto bindings_per_profile = engine_->getNbIOTensors() / num_profiles;
        auto profiles_idx = context_->getOptimizationProfile();
        size_t bindings_size = bindings.size();

        for (auto iter = bindings.begin(); iter != bindings.end(); iter++) {
            // auto binding_index = engine_->getBindingIndex(iter->first.c_str()) +
            //                      profiles_idx * bindings_per_profile;
            std::string name = iter->first;
            // infer_bindings[name.c_str()] = iter->second->get_buffer();
            context_->setTensorAddress(name.c_str(), iter->second->get_buffer());
            LOG_IF(ERROR,
                   is_none(name.c_str()))
                    << "binding name : "
                    << name << " is nullptr";
        }
        cudaEventSynchronize(*event);
        auto result = context_->enqueueV3(stream);
        return result;
    }

    size_t get_device_memory_size() {
        size_t size = m_size;
        return (size >> 8U);
    }

    void set_device_ptr(uint8_t *ptr, bool force = false) {
        if (force || !device_memory_setted_) {
            context_->setDeviceMemory(ptr);
            device_memory_setted_ = true;
        }
    }

    int32_t get_input_num() {
        int32_t input_num = 0;
        for (int i = 0; i < engine_->getNbIOTensors(); i++) {
            auto name = engine_->getIOTensorName(i);
            if (engine_->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT) {
                input_num++;
            }
        }
        return input_num;
    }

    int32_t get_output_num() {
        int32_t output_num = 0;
        for (int i = 0; i < engine_->getNbIOTensors(); i++) {
            auto name = engine_->getIOTensorName(i);
            if (is_output(name)) {
                output_num++;
            }
        }
        return output_num;
    }

    bool check_status() { return runtime_ && engine_ && context_; }

    bool calc_output_shape(const std::unordered_map<std::string, std::vector<int32_t>> &input_info,
                           std::unordered_map<std::string, std::vector<int32_t>> &output_info) {
        if (!check_status()) { return false; }

        auto profile_num = engine_->getNbOptimizationProfiles();
        auto binding_nums = engine_->getNbIOTensors() / profile_num;

        for (int32_t profile_index = 0; profile_index < profile_num; profile_index++) {
            auto fit_flag = true;
            for (auto iter = input_info.begin(); iter != input_info.end(); iter++) {
                // auto binding_index = engine_->getBindingIndex(iter->first.c_str()) + profile_index * binding_nums;
                auto name = iter->first.c_str();
                if (engine_->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT) {
                    auto &input_dims = iter->second;
                    auto max_dim = engine_->getProfileShape(name, profile_index, nvinfer1::OptProfileSelector::kMAX);
                    auto min_dim = engine_->getProfileShape(name, profile_index, nvinfer1::OptProfileSelector::kMIN);
                    if (is_dim_fit(input_dims, max_dim, min_dim)) {
                        if (context_->getOptimizationProfile() != profile_index) {
                            context_->setOptimizationProfileAsync(profile_num, 0);
                        }
                        nvinfer1::Dims dim{};
                        dim.nbDims = input_dims.size();
                        for (int32_t i = 0; i < input_dims.size(); i++) {
                            dim.d[i] = input_dims[i];
                        }
                        context_->setInputShape(name, dim);
                    } else {
                        fit_flag = false;
                        break;
                    }
                } else {
                    LOG(WARNING) << "input name: " << iter->first << " is not input.";
                }
            }
            if (fit_flag) {
                for (int32_t i = 0; i < binding_nums; i++) {
                    auto binding_index = i + profile_index * binding_nums;
                    auto name = m_tensor_names[i];
                    if (!(engine_->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT)) {
                        // std::string name(engine_->getIOTensorName(binding_index));
                        auto dim = context_->getTensorShape(name);
                        std::vector<int32_t> output_dim(dim.nbDims, 0);
                        for (int n = 0; n < dim.nbDims; n++) {
                            output_dim[n] = dim.d[n];
                            if (dim.d[n] <= 0) {
                                LOG(ERROR) << "calc output dimension error. name: " << name << ". dim: " << n << ". value: " << dim.d[n];
                                return false;
                            }
                        }
                        output_info[name] = output_dim;
                    }
                }
                return true;
            }
        }
        return false;
    }

    std::string get_binding_name(int32_t index) {
        if (check_status() && index < engine_->getNbIOTensors()) {
            std::string name(m_tensor_names[index]);
            return name;
        } else {
            return "";
        }
    }

    std::array<std::vector<int32_t>, 2>
    get_binding_dims(const std::string &binding_name) {
        std::array<std::vector<int32_t>, 2> result{};
        auto profile_num = engine_->getNbOptimizationProfiles();
        auto binding_nums = engine_->getNbIOTensors() / profile_num;
        for (int32_t profile_index = 0; profile_index < profile_num; profile_index++) {
            // auto binding_index = engine_->getBindingIndex(binding_name.c_str());
            if (!is_none(binding_name.c_str())) {
                auto max_dim = engine_->getProfileShape(binding_name.c_str(),
                                                        profile_index, nvinfer1::OptProfileSelector::kMAX);
                auto min_dim = engine_->getProfileShape(binding_name.c_str(),
                                                        profile_index, nvinfer1::OptProfileSelector::kMIN);
                for (auto i = 0; i < max_dim.nbDims; i++) {
                    result[0].emplace_back(min_dim.d[i]);
                    result[1].emplace_back(max_dim.d[i]);
                }
                break;
            }
        }
        return result;
    }

private:
    bool is_input(const char *name) {
        return engine_->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT;
    }

    bool is_output(const char *name) {
        return engine_->getTensorIOMode(name) == nvinfer1::TensorIOMode::kOUTPUT;
    }

    bool is_none(const char *name) {
        return engine_->getTensorIOMode(name) == nvinfer1::TensorIOMode::kNONE;
    }

    bool is_dim_fit(const std::vector<int32_t> &input_dim, const nvinfer1::Dims &max, const nvinfer1::Dims &min) {
        if (input_dim.size() != max.nbDims) {
            VLOG(1) << "input dim " << input_dim.size() << " is not fit to engine " << max.nbDims;
            return false;
        }
        if (max.nbDims != min.nbDims) {
            VLOG(1) << "engine max dim " << max.nbDims << " is not fit to engine min dim " << min.nbDims;
            return false;
        }
        for (int32_t i = 0; i < max.nbDims; i++) {
            if (input_dim[i] > max.d[i] || input_dim[i] < min.d[i]) {
                VLOG(1) << "input dim " << i << "'s value : " << input_dim[i] << " is not in range ["
                        << min.d[i] << "~" << max.d[i] << "].";
                return false;
            }
        }
        return true;
    }

    nvinfer1::IRuntime *runtime_ = nullptr;
    nvinfer1::ICudaEngine *engine_ = nullptr;
    nvinfer1::IExecutionContext *context_ = nullptr;
    std::unordered_map<std::string, void *> bindings_{};
    bool device_memory_setted_ = false;
    std::vector<const char *> m_tensor_names{};
    size_t m_size = 0;
};

/** trt implement */
class executor::impl {
public:
    impl() { cudaEventCreate(&consume_); }

    ~impl() {
        if (context_) {
            trt_logger.log(nvinfer1::ILogger::Severity::kVERBOSE, ("context map is: " + std::to_string((int64_t) context_)).c_str());
            delete context_;
        }
        if (engine_) {
            // engine_->destroy();
            delete engine_;
        }
        if (runtime_) {
            // runtime_->destroy();
            delete runtime_;
        }
        if (consume_) {
            cudaEventDestroy(consume_);
        }
    }

    bool load(const std::string &path, bool alloc = true) {
        std::vector<uint8_t> model{};
        std::fstream file(path, std::ios::in | std::ios::binary);
        if (file.is_open()) {
            auto rdbuf = file.rdbuf();
            model.resize(rdbuf->pubseekoff(std::ios::beg, std::ios::end, std::ios::in));
            rdbuf->pubseekpos(std::ios::beg, std::ios::in);
            rdbuf->sgetn((char *) model.data(), model.size());
            file.close();
            return load(model.data(), model.size(), alloc);
        } else {
            trt_logger.log(nvinfer1::ILogger::Severity::kERROR, "can not open model file");
            return false;
        }
    }

    bool load(const uint8_t *data, size_t length, bool alloc = true) {
        self_alloc_ = alloc;
        models_.emplace_back(std::make_unique<nv_model>(data, length));
        if (models_.back()->check_status()) {
            if (!models_.empty() && models_.back()) {
                size_t size = models_.back()->get_device_memory_size();
                trt_logger.log(nvinfer1::ILogger::Severity::kINFO, ("model needs " + std::to_string(size) + " bytes memory").c_str());
                return true;
            } else {
                trt_logger.log(nvinfer1::ILogger::Severity::kERROR, "can not deserialize cuda engine");
                return false;
            }
        } else {
            models_.pop_back();
            return false;
        }
    }

    void set_device_memory(void *ptr) { extern_buffer_ = (uint8_t *) ptr; }
    size_t get_device_memory() {
        size_t size = 0;
        for (const auto &model : models_) {
            if (model->get_device_memory_size() > size) {
                size = model->get_device_memory_size();
            }
        }
        return size;
    }

    void release() { models_.clear(); }

    int get_model_num() { return models_.size(); }
    int get_input_num(int32_t index) { return index >= models_.size() ? 0 : models_[index]->get_input_num(); }
    int get_output_num(int32_t index) { return index >= models_.size() ? 0 : models_[index]->get_output_num(); }
    std::string get_binding_name(int32_t model_idx, int32_t binding_idx) {
        if (model_idx >= models_.size()) {
            LOG(WARNING) << "model index " << model_idx << " is larger than model size " << models_.size();
            return "";
        }
        return models_[model_idx]->get_binding_name(binding_idx);
    };

    std::array<std::vector<int32_t>, 2>
    get_binding_dims(int32_t model_idx, const std::string &binding_name) {
        std::array<std::vector<int32_t>, 2> results{};
        if (model_idx >= models_.size()) {
            return results;
        } else {
            return models_[model_idx]->get_binding_dims(binding_name);
        }
    }

    std::unordered_map<std::string, std::vector<int32_t>>
    get_output_shapes(const std::unordered_map<std::string, std::vector<int32_t>> &input_info) {
        std::unordered_map<std::string, std::vector<int32_t>> empty_result{};
        for (auto &model : models_) {
            if (model->calc_output_shape(input_info, empty_result)) {
                return empty_result;
            }
        }
        return empty_result;
    }

public:
    template<typename T>
    int execute_dynamic(const std::vector<const gpu_tensor<T> *> &bindings, cudaStream_t stream) {
        int32_t model_index = -1;
        for (int32_t n = 0; n < models_.size(); n++) {
            bool fit = true;
            for (size_t i = 0; i < bindings.size(); i++) {
                fit &= models_[n]->is_fit(i, bindings[i]);
            }
            if (fit) {
                model_index = n;
                break;
            }
        }
        if (model_index != -1) {
            if (self_alloc_) {
                auto size = models_[model_index]->get_device_memory_size();
                if (!engine_buffer_) {
                    size = (size == 0) ? 1 : size;
                    engine_buffer_ = std::make_unique<gpu_tensor<uint8_t>>(1, 1, 256, size);
                }
                if (size > engine_buffer_->get_width()) {
                    engine_buffer_->resize(1, 1, 256, size);
                }
                models_[model_index]->set_device_ptr(engine_buffer_->get_buffer());
            } else {
                models_[model_index]->set_device_ptr(extern_buffer_, true);
            }
            auto status = cudaEventSynchronize(consume_);
            return models_[model_index]->enqueue(bindings, stream, &consume_);
        } else {
            LOG(ERROR) << "model input size cannot fit input tensor size";
            return false;
        }
    }

    template<typename T>
    int execute_dynamic(std::unordered_map<std::string, const gpu_tensor<T> *> bindings, cudaStream_t stream) {
        int32_t model_index = -1;
        for (int32_t n = 0; n < models_.size(); n++) {
            bool fit = true;
            for (auto iter = bindings.begin(); iter != bindings.end(); iter++) {
                fit &= models_[n]->is_fit(iter->first, iter->second);
            }
            if (fit) {
                model_index = n;
                break;
            }
        }
        if (model_index != -1) {
            if (self_alloc_) {
                auto size = models_[model_index]->get_device_memory_size();
                if (!engine_buffer_) {
                    size = (size == 0) ? 1 : size;
                    engine_buffer_ = std::make_unique<gpu_tensor<uint8_t>>(1, 1, 256, size);
                }
                if (size > engine_buffer_->get_width()) {
                    engine_buffer_->resize(1, 1, 256, size);
                }
                models_[model_index]->set_device_ptr(engine_buffer_->get_buffer());
            } else {
                models_[model_index]->set_device_ptr(extern_buffer_, true);
            }
            auto status = cudaEventSynchronize(consume_);
            return models_[model_index]->enqueue(bindings, stream, &consume_);
        } else {
            LOG(ERROR) << "model input size cannot fit input tensor size";
            return false;
        }
    }

private:
    std::vector<std::unique_ptr<nv_model>> models_{};

    nvinfer1::IRuntime *runtime_ = nullptr;
    nvinfer1::ICudaEngine *engine_ = nullptr;
    nvinfer1::IExecutionContext *context_ = nullptr;
    cudaEvent_t consume_ = nullptr;

    std::unique_ptr<gpu_tensor<uint8_t>> engine_buffer_ = nullptr;

    bool self_alloc_ = true;
    uint8_t *extern_buffer_ = nullptr;
};

executor::executor() { ptr_ = std::make_unique<impl>(); }
executor::executor(const std::string &path) : executor() { ptr_->load(path); }
executor::executor(const uint8_t *data, size_t length) : executor() { ptr_->load(data, length); }
executor::~executor() = default;

bool executor::load(const std::string &path, bool alloc) { return ptr_->load(path, alloc); }
bool executor::load(const uint8_t *data, size_t length, bool alloc) { return ptr_->load(data, length, alloc); }

void executor::set_device_memory(void *ptr) { ptr_->set_device_memory(ptr); }
size_t executor::get_device_memory() { return ptr_->get_device_memory(); }

int executor::get_model_num() { return ptr_->get_model_num(); }
int executor::get_input_num(int32_t index) { return ptr_->get_input_num(index); }
int executor::get_output_num(int32_t index) { return ptr_->get_output_num(index); }
std::string executor::get_binding_name(int32_t model_idx, int32_t binding_idx) {
    return ptr_->get_binding_name(model_idx, binding_idx);
}

std::array<std::vector<int32_t>, 2>
executor::get_binding_dim(int32_t model_idx, const std::string &binding_name) {
    return ptr_->get_binding_dims(model_idx, binding_name);
}

std::unordered_map<std::string, std::vector<int32_t>>
executor::get_output_shapes(std::unordered_map<std::string, std::vector<int32_t>> input_info) {
    return ptr_->get_output_shapes(input_info);
}

int executor::execute_dynamic(const std::vector<const gpu_tensor<float> *> &bindings, int64_t stream) {
    return ptr_->execute_dynamic<float>(bindings, (cudaStream_t) stream);
}

int executor::execute_dynamic(const std::vector<const gpu_tensor<half> *> &bindings, int64_t stream) {
    return ptr_->execute_dynamic<half>(bindings, (cudaStream_t) stream);
}

int executor::execute_dynamic(std::unordered_map<std::string, const gpu_tensor<float> *> bindings, int64_t stream) {
    return ptr_->execute_dynamic<float>(bindings, (cudaStream_t) stream);
}

int executor::execute_dynamic(std::unordered_map<std::string, const gpu_tensor<half> *> bindings, int64_t stream) {
    return ptr_->execute_dynamic<half>(bindings, (cudaStream_t) stream);
}
}// namespace gpu
