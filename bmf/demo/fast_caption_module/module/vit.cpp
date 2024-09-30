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
#include <chrono>
#include <condition_variable>
#include <future>
#include <opencv2/opencv.hpp>
#include <sys/prctl.h>
#include <fstream>
#include <thread>
#include <vector>
#include <string>
#include <regex>

#include <bmf/sdk/video_frame.h>
#include <bmf/sdk/module.h>
#include <bmf/sdk/module_registry.h>
#include <bmf/sdk/ffmpeg_helper.h>
#include <bmf/sdk/bmf_av_packet.h>

#include <cuda_runtime_api.h>
#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <NvInferPluginUtils.h>
#include <NvInferRuntime.h>

// torch
#include <torch/torch.h>
#include <ATen/ATen.h>

// trt
#include <executor.h>
#include <gpu_tensor.h>

// include
#include "image.h"
#include "vit.h"
#include "common.h"

using namespace nvinfer1;
using json = nlohmann::json;
#define IMG_MAX_SIZE 10000000

vit::vit(vit_config &config, int32_t gpu_id)
{
    cudaSetDevice(gpu_id);
    m_gpu_id = gpu_id;
    m_running = true;
    // config
    m_hidden_size = config.hidden_size;
    m_numframes = config.num_frames;
    std::string vit_path = config.path + "/vit.trt", image_newline_path = config.path + "/image_newline_c++.pth", embed_table_path = config.path + "/table.raw";
    std::vector<half> f(m_hidden_size);
    load(image_newline_path, f);
    std::vector<half> f2((m_vocab_size + 2) * m_hidden_size);
    load(embed_table_path, f2);
    m_image_newline = at::from_blob(f.data(), f.size(), at::TensorOptions().dtype(at::kHalf)).clone();
    auto max_channel = config.num_patches / config.downsample;
    m_grid_pinpoints = config.grid_pinpoints;
    m_frame_pool = std::vector<unsigned char>(IMG_MAX_SIZE * m_numframes * 10);

    int32_t pre_ids_size = config.pre_ids.size();
    int32_t post_ids_size = config.post_ids.size();
    std::vector<half> pre_tokens(pre_ids_size * m_hidden_size);
    std::vector<half> post_tokens(post_ids_size * m_hidden_size);
    for (size_t i = 0; i < pre_ids_size; i++)
    {
        memcpy(pre_tokens.data() + i * m_hidden_size, f2.data() + config.pre_ids[i] * m_hidden_size, m_hidden_size * 2);
    }
    for (size_t i = 0; i < post_ids_size; i++)
    {
        memcpy(post_tokens.data() + i * m_hidden_size, f2.data() + config.post_ids[i] * m_hidden_size, m_hidden_size * 2);
    }
    m_pre_tokens = at::from_blob(pre_tokens.data(), {pre_ids_size, m_hidden_size}, at::TensorOptions().dtype(at::kHalf)).clone();
    m_post_tokens = at::from_blob(post_tokens.data(), {post_ids_size, m_hidden_size}, at::TensorOptions().dtype(at::kHalf)).clone();
    m_engine = std::make_unique<gpu::executor>();
    if (!m_engine->load(vit_path))
    {
        LOG(ERROR) << "load model error. " << vit_path;
    }
    m_vit_input = std::make_unique<gpu::gpu_tensor<half>>(config.max_batch, 3, config.image_size, config.image_size);
    m_vit_output = std::make_unique<gpu::gpu_tensor<half>>(1, config.max_batch, max_channel * max_channel, config.hidden_size);
    inference_binding = {m_vit_input.get(), m_vit_output.get()};

    // stream and event
    cudaStreamCreate(&m_stream);
    auto event_flag = cudaEventDisableTiming | cudaEventBlockingSync;
    cudaEventCreateWithFlags(&m_event, event_flag);
    // thread
    m_process_thread = std::thread([&]
                                   { process(); });
}

void vit::send(std::unique_ptr<vit_task> task)
{
    auto future = cpu_executor<std::unique_ptr<vit_task>>::instance()->exec(
        [&](std::unique_ptr<vit_task> task)
        { return preprocess(std::move(task)); },
        std::move(task));
    std::lock_guard<std::mutex> guard(m_work_mutex);
    m_work_queue.emplace(std::move(future));
}

std::future<std::unique_ptr<vit_task>> vit::receive()
{
    std::unique_lock<std::mutex> lock(m_vit_mutex);
    m_vit_cv.wait(lock, [&]
                  { return !m_receive_queue.empty(); });
    auto vit_output_future = std::move(m_receive_queue.front());
    vit_output_future.wait();
    m_receive_queue.pop();
    lock.unlock();
    return vit_output_future;
}

std::unique_ptr<vit_task> vit::preprocess(std::unique_ptr<vit_task> &&task)
{
    auto idx = task->get_index();
    auto frame_list = task->get_vframes();
    std::vector<float> new_img_data;
    process_image(frame_list, new_img_data, idx);
    auto size = new_img_data.size();
    // write
    half *vit_data = task->get_buffer();
    for (size_t i = 0; i < size; i++)
    {
        vit_data[i] = __float2half(new_img_data[i]);
    }
    int32_t batch = size / m_image_size / m_image_size / 3;
    auto frame = frame_list[0].frame();
    int32_t height = frame.height();
    int32_t width = frame.width();
    task->set_image_size(height, width, batch);
    auto input_tensor = at::zeros({batch, m_num_patches * m_num_patches, m_hidden_size}, at::kHalf);
    task->set_input_tensor(std::move(input_tensor));

    return std::move(task);
}

std::unique_ptr<vit_task> vit::gpuprocess(std::unique_ptr<vit_task> &&task)
{
    auto tensor_size = task->get_image_size();
    int32_t batch = tensor_size.batch;
    auto patch_size = m_num_patches * m_num_patches;
    m_vit_input->resize(batch, 3, m_image_size, m_image_size);
    m_vit_output->resize(1, batch, patch_size, m_hidden_size);
    cudaMemcpyAsync(m_vit_input->get_buffer(), task->get_buffer(), batch * 3 * m_image_size * m_image_size * sizeof(half), cudaMemcpyDefault, m_stream);
    if (!m_engine->execute_dynamic(inference_binding, (int64_t)m_stream))
    {
        LOG(ERROR) << "vit inference failed";
    }
    auto vit_postprocess_input_cpu = task->get_postprocess_input_tensor();
    cudaMemcpyAsync(vit_postprocess_input_cpu.data_ptr(), m_vit_output->get_buffer(), batch * patch_size * m_hidden_size * sizeof(half), cudaMemcpyDefault, m_stream);
    task->record(m_stream);
    return std::move(task);
}

void vit::process()
{
    cudaSetDevice(m_gpu_id);
    while (m_running)
    {
        std::unique_lock<std::mutex> lock(m_work_mutex);
        if (m_work_queue.empty())
        {
            lock.unlock();
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }

        auto future = std::move(m_work_queue.front());
        m_work_queue.pop();
        lock.unlock();
        future.wait();

        cudaStreamWaitEvent(m_stream, m_event);
        auto task = vit::gpuprocess(std::move(future.get()));
        cudaEventRecord(m_event, m_stream);
        auto gpu_future = cpu_executor<std::unique_ptr<vit_task>>::instance()->exec(
            [&](std::unique_ptr<vit_task> task)
            { return postprocess(std::move(task)); },
            std::move(task));
        std::lock_guard<std::mutex> guard(m_vit_mutex);
        m_receive_queue.emplace(std::move(gpu_future));
        m_vit_cv.notify_one();
    }
}

std::unique_ptr<vit_task> vit::postprocess(std::unique_ptr<vit_task> &&task)
{
    task->wait();
    auto tensor = task->get_postprocess_input_tensor();
    auto vit_size = task->get_image_size();
    int32_t height = vit_size.height, width = vit_size.width, batch = vit_size.batch;
    at::Tensor new_image_feature = vit_image_feature(tensor, height, width, batch / m_numframes);
    task->set_image_feature(std::move(new_image_feature));

    return std::move(task);
}