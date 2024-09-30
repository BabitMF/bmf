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
#ifndef _VIT_H
#define _VIT_H

#include <cuda_runtime_api.h>
#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <NvInferPluginUtils.h>
#include <NvInferRuntime.h>

// torch
#include <torch/torch.h>
#include <ATen/ATen.h>

// trt
#include <cuda_fp16.h>
#include "gpu_tensor.h"
#include "executor.h"

// include
#include "image.h"
#include "common.h"
typedef struct
{
    int32_t height, width, batch;
} vit_tensor_size;

typedef struct
{
    vit_tensor_size vit_size;
    at::Tensor tensor;
} vit_data;

template <typename T>
class host_buffer
{
public:
    host_buffer() = delete;
    explicit host_buffer(size_t element_count)
    {
        constexpr size_t align = 256;
        m_length = (element_count * sizeof(T) + align - 1) / align * align;

        auto status = cudaMallocHost((void **)&m_buf, m_length);
        if (status != cudaSuccess)
        {
            LOG(ERROR) << "Apply host buf error. length is: " << m_length << ". "
                       << cudaGetErrorName(status) << ":" << cudaGetErrorString(status);
            m_buf = nullptr;
        }
    }
    T *buffer() { return (T *)m_buf; }
    const T *buffer() const { return (T *)m_buf; }

    size_t length() const { return m_length; }

    ~host_buffer()
    {
        if (m_buf)
        {
            cudaFreeHost(m_buf);
        }
    }

private:
    size_t m_length = 0;
    void *m_buf = nullptr;
};

template <typename T, class pool_t>
class host_tensor
{
public:
    explicit host_tensor(int32_t width, int32_t height, int32_t channel, pool_t *pool)
        : m_width(width), m_height(height), m_channel(channel)
    {
        size_t size = (size_t)width * (size_t)height * (size_t)channel;
        if (pool)
        {
            m_pool = pool;
            m_buffer = pool->get_obj();
            if (m_buffer->length() < size * sizeof(T))
            {
                LOG(ERROR) << "tensor buffer" << m_buffer->length()
                           << " cannot hold tensor size " << size;
            }
        }
        else
        {
            m_buffer = std::make_unique<host_buffer<T>>(size);
        }
    }

    ~host_tensor()
    {
        if (m_pool && m_buffer)
        {
            m_pool->ret_obj(std::move(m_buffer));
        }
    }

    int32_t get_w() const { return m_width; }
    int32_t get_h() const { return m_height; }
    int32_t get_c() const { return m_channel; }

    T *buffer() { return m_buffer->buffer(); }
    const T *buffer() const { return m_buffer->buffer(); }

private:
    int32_t m_width = 0;
    int32_t m_height = 0;
    int32_t m_channel = 0;

    pool_t *m_pool = nullptr;
    std::unique_ptr<host_buffer<T>> m_buffer = nullptr;
};

template <class buffer, typename... args>
class buffer_pool
{
    using creator_t = std::function<std::unique_ptr<buffer>(void)>;

public:
    explicit buffer_pool(int32_t num, args... arg)
    {
        m_creator = std::bind(
            [](args... arg) -> std::unique_ptr<buffer>
            {
                return std::make_unique<buffer>(std::forward<args...>(arg...));
            },
            std::forward<args>(arg)...);
        for (int32_t n = 0; n < num; n++)
        {
            m_queue.emplace(m_creator());
        }
    }

    std::unique_ptr<buffer> get_obj()
    {
        std::lock_guard<std::mutex> guard(m_mutex);
        if (m_queue.empty())
        {
            LOG(WARNING) << "pool is empty, alloc new";
            return m_creator();
        }
        else
        {
            auto obj = std::move(m_queue.front());
            m_queue.pop();
            return obj;
        }
        return nullptr;
    }

    void ret_obj(std::unique_ptr<buffer> &&obj)
    {
        if (obj)
        {
            std::lock_guard<std::mutex> guard(m_mutex);
            m_queue.emplace(std::move(obj));
        }
    }

private:
    creator_t m_creator;
    std::queue<std::unique_ptr<buffer>> m_queue;
    std::mutex m_mutex{};
    std::tuple<args...> m_arg;
};

template <typename T, class pool_t>
class async_task
{
public:
    explicit async_task(std::vector<bmf_sdk::VideoFrame> &&vframes, int32_t index) {
        m_vframes = std::move(vframes);
        m_idx = index;
    }

    ~async_task()
    {
        if (m_event)
        {
            cudaEventDestroy(m_event);
            m_event = nullptr;
        }
    }

    void set_dim(int32_t width, int32_t height, int32_t channel, pool_t *pool)
    {
        if (pool)
        {
            m_pool = std::make_unique<host_tensor<T, pool_t>>(width, height, channel, pool);
        }
        else
        {
            m_pool = nullptr;
        }
    }

    void set_image_size(int32_t height, int32_t width, int32_t batch)
    {
        m_image_size = {.height = height, .width = width, .batch = batch};
    }

    void set_input_tensor(at::Tensor &&tensor)
    {
        m_tensor = std::move(tensor);
    }

    void set_image_feature(at::Tensor &&new_image_feature)
    {
        m_image_feature = std::move(new_image_feature);
    }

    inline int32_t get_index() const
    {
        return m_idx;
    }

    std::vector<bmf_sdk::VideoFrame> get_vframes() {
        return m_vframes;
    }

    void record(cudaStream_t stream = 0)
    {
        cudaEventCreateWithFlags(&m_event,
                                 cudaEventBlockingSync | cudaEventDisableTiming);
        cudaEventRecord(m_event, stream);
    }

    void wait()
    {
        cudaError_t status = cudaEventQuery(m_event);
        while (status != cudaSuccess)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            status = cudaEventQuery(m_event);
        }
        cudaEventSynchronize(m_event);
    }

    vit_tensor_size get_image_size()
    {
        return m_image_size;
    }

    at::Tensor get_postprocess_input_tensor()
    {
        return m_tensor;
    }

    at::Tensor get_postprocess_output_tensor()
    {
        return m_image_feature;
    }

    T *get_buffer() { return m_pool ? m_pool->buffer() : nullptr; }

private:
    std::unique_ptr<host_tensor<T, pool_t>> m_pool = nullptr;
    bmf_sdk::BMFAVPacket m_pkt;
    std::vector<bmf_sdk::VideoFrame> m_vframes; 
    int32_t m_idx;
    vit_tensor_size m_image_size;
    at::Tensor m_tensor;
    at::Tensor m_image_feature;
    cudaEvent_t m_event = nullptr;
};
using cpu_pool = buffer_pool<host_buffer<half>, size_t>;
using vit_task = async_task<half, cpu_pool>;
typedef struct
{
    int32_t image_size = 336;
    int32_t num_frames = 1;
    int32_t patch_size = 14;
    int32_t num_patches = 24;
    int32_t downsample = 1;
    int32_t hidden_size = 3584;
    int32_t max_batch = 40;
    std::string path;
    std::vector<std::vector<int32_t>> grid_pinpoints;
    std::vector<int32_t> pre_ids;
    std::vector<int32_t> post_ids;
} vit_config;

class vit
{
public:
    vit() {}
    explicit vit(vit_config &config, int32_t gpu_id);
    ~vit()
    {
        m_running = false;
        if (m_process_thread.joinable())
        {
            m_process_thread.join();
        }
        cudaStreamDestroy(m_stream);
        m_grid_pinpoints.clear();
        m_frame_pool.clear();

        m_engine = nullptr;
        m_vit_input = nullptr;
        m_vit_output = nullptr;
    }
    at::Tensor postprocess(vit_data data);

    std::future<std::unique_ptr<vit_task>> receive();
    void process();

    std::unique_ptr<vit_task> preprocess(std::unique_ptr<vit_task> &&task);
    std::unique_ptr<vit_task> gpuprocess(std::unique_ptr<vit_task> &&task);
    std::unique_ptr<vit_task> postprocess(std::unique_ptr<vit_task> &&task);
    void send(std::unique_ptr<vit_task> task);

    void reset_frame_pool()
    {
        memset(m_frame_pool.data(), 0, m_frame_pool.size());
    }

private:
    at::Tensor vit_image_feature(at::Tensor &vit_postprocess_input_cpu, int32_t height, int32_t width, int32_t batch)
    {
        auto size = select_best_resolution(height, width) / m_image_size;
        auto patch_size = m_num_patches;
        std::vector<at::Tensor> image_feature_list;
        for (size_t i = 0; i < m_numframes; i++)
        {
            // m_grid_pinpoints
            auto base_image_feature = vit_postprocess_input_cpu[i * batch];
            auto image_feature = vit_postprocess_input_cpu.slice(0, i * batch + 1, (i + 1) * batch);
            image_feature = image_feature.view({size.height, size.width, patch_size, patch_size, -1});
            image_feature = image_feature.permute({4, 0, 2, 1, 3}).contiguous().flatten(1, 2).flatten(2, 3);
            auto image_sizes = image_feature.sizes();

            int32_t current_height = image_sizes.data()[1], current_width = image_sizes.data()[2];
            float original_ratio = 1.f * width / height;
            float data_ratio = 1.f * current_width / current_height;
            if (original_ratio > data_ratio)
            {
                float scale_factor = 1.f * current_width / width;
                int32_t new_height = int32_t(height * scale_factor);
                int32_t padding = (current_height - new_height) / 2;
                image_feature = image_feature.index({torch::indexing::Slice(), torch::indexing::Slice(
                                                     padding, current_height - padding), torch::indexing::Slice()}).contiguous();
            }
            else
            {
                float scale_factor = 1.f * current_height / height;
                int32_t new_width = int32_t(width * scale_factor);
                int32_t padding = (current_width - new_width) / 2;
                image_feature = image_feature.index({"...", torch::indexing::Slice(
                                                    padding, current_width - padding, 1)}).contiguous();
            }
            image_sizes = image_feature.sizes();
            auto image_newline_size = m_image_newline.sizes().data();
            image_feature = torch::cat({image_feature, m_image_newline.view({m_image_newline.sizes().data()[0], 1, 1}).expand({image_sizes.data()[0], image_sizes.data()[1], 1})}, -1);
            image_feature = image_feature.flatten(1, 2).transpose(0, 1);
            image_feature = torch::cat({base_image_feature, image_feature}, 0);
            image_feature_list.emplace_back(image_feature);
        }
        auto new_image_feature = torch::cat(image_feature_list, 0);
        // embed token add
        new_image_feature = torch::cat({m_pre_tokens, new_image_feature, m_post_tokens}, 0);
        return new_image_feature;
    }

    std::vector<cv::Mat> divide_patches(cv::Mat &image, int32_t patch_size)
    {
        auto width = image.cols;
        auto height = image.rows;
        std::vector<cv::Mat> patch_list;
        for (size_t i = 0; i < height; i += patch_size)
        {
            for (size_t j = 0; j < width; j += patch_size)
            {
                cv::Rect area = cv::Rect(j, i, patch_size, patch_size);
                patch_list.emplace_back(image(area));
            }
        }
        return patch_list;
    }

    cv::Size select_best_resolution(int32_t height, int32_t width)
    {
        int32_t max_effective_resolution = 0;
        int32_t min_wasted_resolution = INT_MAX;
        cv::Size best_fit;
        for (auto size : m_grid_pinpoints)
        {
            float scale = std::min((float)size[0] / width, (float)size[1] / height);
            int32_t downscale_width = scale * width, downscale_height = scale * height;
            int32_t effective_resolution = std::min(downscale_width * downscale_height, height * width);
            int32_t wasted_resolution = size[0] * size[1] - effective_resolution;

            if (effective_resolution > max_effective_resolution || ((effective_resolution == max_effective_resolution) && (wasted_resolution < min_wasted_resolution)))
            {
                max_effective_resolution = effective_resolution;
                min_wasted_resolution = wasted_resolution;
                best_fit = {size[0], size[1]};
            }
        }
        return best_fit;
    }

    void resize_and_pad_image(cv::Mat &rgb, cv::Mat &new_img, cv::Size target_size)
    {
        int32_t original_width = rgb.cols;
        int32_t original_height = rgb.rows;
        float scale_w = target_size.width / (float)original_width;
        float scale_h = target_size.height / (float)original_height;
        cv::Size new_size;
        if (scale_w < scale_h)
        {
            new_size.width = target_size.width;
            new_size.height = std::min(target_size.height, (int32_t)(std::ceil(scale_w * original_height)));
        }
        else
        {
            new_size.height = target_size.height;
            new_size.width = std::min(target_size.width, (int32_t)(std::ceil(scale_h * original_width)));
        }
        cv::Mat resized_img;
        cv::resize(rgb, resized_img, new_size);
        int32_t pad_x = (target_size.width - new_size.width) / 2;
        int32_t pad_y = (target_size.height - new_size.height) / 2;
        cv::copyMakeBorder(resized_img, new_img, pad_y, target_size.height - new_size.height - pad_y,
                           pad_x, target_size.width - pad_x - new_size.width, cv::BORDER_CONSTANT, cv::Scalar({0, 0, 0}));
    }

    void process_image(std::vector<bmf_sdk::VideoFrame> &frame_list, std::vector<float> &dst, int32_t idx)
    {
        float *dst_ptr = dst.data();
        for (auto &vframe : frame_list)
        {
            auto frame = vframe.frame();
            auto width = frame.width();
            auto height = frame.height();
            auto best_resolution = select_best_resolution(height, width);
            auto data_ptr = frame.plane(0).data<uint8_t>();
            cv::Mat rgb = cv::Mat(cv::Size(width, height), CV_8UC3, data_ptr, frame.plane(0).stride(0));
            cv::Mat image_padded = cv::Mat(best_resolution, CV_8UC3);
            resize_and_pad_image(rgb, image_padded, best_resolution);
            auto padding_list = divide_patches(image_padded, m_image_size);
            cv::Mat image_original_resize;
            cv::resize(rgb, image_original_resize, cv::Size(m_image_size, m_image_size));
            padding_list.insert(padding_list.begin(), image_original_resize);
            auto padding_list_size = padding_list.size() * m_image_size * m_image_size * 3 * m_numframes;
            if (dst.size() != padding_list_size)
            {
                dst.resize(padding_list_size);
                dst_ptr = dst.data();
            }
            cv::Scalar mean = {0.48145466f, 0.4578275f, 0.40821073f};
            cv::Scalar std = {0.26862954f, 0.26130258f, 0.27577711f};
            std::vector<cv::Mat> image_channels;
            for (size_t i = 0; i < padding_list.size(); i++)
            {
                std::vector<cv::Mat> rgbchannels(3);
                cv::split(padding_list[i], rgbchannels);
                for (size_t j = 0; j < 3; j++)
                {
                    cv::Mat temp_img;
                    rgbchannels[j].convertTo(temp_img, CV_32FC1, 1.f / (255.f * std[j]), -mean[j] / std[j]);
                    memcpy(dst_ptr + (i * 3 + j) * m_image_size * m_image_size, temp_img.data, m_image_size * m_image_size * 4);
                }
            }

            dst_ptr += padding_list.size() * m_image_size * m_image_size * 3;
        }
    }

    bool m_running = false;
    cudaStream_t m_stream = nullptr;
    cudaEvent_t m_event = nullptr;
    int32_t m_gpu_id = 0;
    int32_t m_numframes = 1;
    int32_t m_image_size = 336;
    int32_t m_patch_size = 14;
    int32_t m_num_patches = 24;
    int32_t m_downsample = 1;
    int32_t m_vocab_size = 32000;
    int32_t m_hidden_size = 4096;

    at::Tensor m_image_newline;
    std::vector<std::vector<int32_t>> m_grid_pinpoints;
    std::vector<unsigned char> m_frame_pool;
    at::Tensor m_pre_tokens;
    at::Tensor m_post_tokens;
    std::unique_ptr<gpu::executor> m_engine = nullptr;
    std::unique_ptr<gpu::gpu_tensor<half>> m_vit_input = nullptr;
    std::unique_ptr<gpu::gpu_tensor<half>> m_vit_output = nullptr;
    std::vector<const gpu::gpu_tensor<half> *> inference_binding;
    std::queue<std::future<std::unique_ptr<vit_task>>> m_work_queue;
    std::queue<std::future<std::unique_ptr<vit_task>>> m_receive_queue;
    std::queue<vit_tensor_size> m_vit_queue;

    std::thread m_process_thread;
    std::condition_variable m_work_cv;
    std::mutex m_work_mutex;
    std::condition_variable m_vit_cv;
    std::mutex m_vit_mutex;
};
#endif