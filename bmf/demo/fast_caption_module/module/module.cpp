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
#include <boost/uuid/uuid.hpp>
#include <boost/uuid/uuid_generators.hpp>
#include <boost/uuid/uuid_io.hpp>

#include <bmf/sdk/video_frame.h>
#include <bmf/sdk/module.h>
#include <bmf/sdk/module_registry.h>
#include <bmf/sdk/ffmpeg_helper.h>
#include <bmf/sdk/bmf_av_packet.h>
#include <bmf/sdk/log.h>

#include <cuda_runtime_api.h>
#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <NvInferPluginUtils.h>
#include <NvInferRuntime.h>

// tensorrt llm include
#include <tensorrt_llm/executor/executor.h>
#include <tensorrt_llm/plugins/api/tllmPlugin.h>
#include "tensorrt_llm/executor/tensor.h"
#include "tensorrt_llm/executor/types.h"

// torch
#include <torch/torch.h>
#include <ATen/ATen.h>

// include
#include "image.h"
#include "common.h"
#include "vit.h"
#include "tokenizer.h"

using namespace nvinfer1;

namespace tlc = tensorrt_llm::common;
namespace tle = tensorrt_llm::executor;

using json = nlohmann::json;
using ordered_json = nlohmann::json_abi_v3_11_2::ordered_json;

class caption : public bmf_sdk::Module
{
public:
    caption(int node_id, bmf_sdk::JsonParam option) : bmf_sdk::Module(node_id, option)
    {
        auto json_value = option.json_value_;
        if (json_value.find("gpu") != json_value.end() && json_value["gpu"].is_number())
        {
            m_gpu_ids = {json_value["gpu"]};
        }

        cudaSetDevice(m_gpu_ids[0]);
        std::string pre_prompts = "[INST] <image>";
        set_number("num_frames", m_numframes, json_value);
        set_number("batch_size", m_batch_size, json_value);
        std::string path, vit_path, kvcache_path, tokenizer_path, llava_config_path, preprocess_config_path;
        set_string("model_path", path, json_value);
        set_string("result_path", result_path, json_value);
        set_string("cap_prompt", cap_prompt, json_value);
        std::string post_prompts = cap_prompt;
        ans_file = std::ofstream(result_path, std::ios::out);
        initTrtLlmPlugins();
        m_num_patches = m_image_size / m_patch_size;
        size_t max_batch = m_numframes * 5;

        vit_path = path + "/vit.trt";
        kvcache_path = path + "/llm_engine";
        tokenizer_path = path + "/tokenizer.json";
        llava_config_path = path + "/config.json";
        preprocess_config_path = path + "/preprocessor_config.json";
        // llava config
        std::ifstream file(llava_config_path);
        json config_json;
        file >> config_json;
        set_number("hidden_size", m_hidden_size, config_json);
        m_grid_pinpoints = std::vector<std::vector<int32_t>>();
        if (config_json.find("image_grid_pinpoints") != config_json.end())
        {
            m_grid_pinpoints = config_json["image_grid_pinpoints"];
        }
        file.close();
        file = std::ifstream(preprocess_config_path);
        file >> config_json;
        set_number("crop_size", m_image_size, config_json);
        m_pool = std::make_unique<cpu_pool>(10, max_batch * 3 *
                                                    m_image_size * m_image_size);
        // myself tokenizer
        m_tokenizer = std::make_unique<tokenizer>(tokenizer_path);
        std::vector<int32_t> m_pre_ids = m_tokenizer->encode("<s> " + pre_prompts, 128);
        std::vector<int32_t> m_post_ids = m_tokenizer->encode("\n" + post_prompts + " [/INST]", 128);
        m_post_ids.insert(m_post_ids.begin(), 28705);
        auto tensor_option = torch::TensorOptions().dtype(at::kInt);

        pre_ids = torch::from_blob(m_pre_ids.data(), {(long)m_pre_ids.size()}, tensor_option).clone();
        post_ids = torch::from_blob(m_post_ids.data(), {(long)m_post_ids.size()}, tensor_option).clone();
        // vit
        vit_config config = {.image_size = m_image_size, .num_frames = m_numframes, .patch_size = m_patch_size, .num_patches = m_num_patches, .downsample = m_downsample, .hidden_size = m_hidden_size, .max_batch = max_batch, .path = path, .grid_pinpoints = m_grid_pinpoints, .pre_ids = m_pre_ids, .post_ids = m_post_ids};
        m_vit = std::make_unique<vit>(config, m_gpu_ids[0]);
        // kvcache
        tle::ParallelConfig parallel_config = tle::ParallelConfig(tle::CommunicationType::kMPI, tle::CommunicationMode::kLEADER,
                                                                  m_gpu_ids, std::nullopt, std::nullopt);
        auto kvcache_config = tle::KvCacheConfig(false, 32768, std::nullopt, std::nullopt, 0.9f, std::nullopt, true);
        m_executor = std::make_unique<tle::Executor>(kvcache_path, tle::ModelType::kDECODER_ONLY,
                                                     tle::ExecutorConfig(1, tle::SchedulerConfig(), kvcache_config,
                                                                         false, true, tle::kDefaultIterStatsMaxIterations, tle::kDefaultRequestStatsMaxIterations,
                                                                         tle::BatchingType::kINFLIGHT, parallel_config));

        m_running = true;
        m_vit_thread = std::thread([&]
                                   { receive(); });
        m_result_thread = std::thread([&]
                                      { write_results_to_file(); });
    }

    ~caption()
    {
        close();
    }

    virtual int init()
    {
        return 0;
    }

    virtual int reset()
    {
        m_input_idx = 0;
        return 0;
    }

    virtual int close()
    {
        m_running = false;
        m_vit = nullptr;
        if (m_vit_thread.joinable())
        {
            m_vit_thread.join();
        }
        m_executor = nullptr;
        if (m_result_thread.joinable())
        {
            m_result_thread.join();
        }
        ans_file.close();
        return 0;
    }

    virtual int process(bmf_sdk::Task &task)
    {
        bmf_sdk::Packet pkt;
        auto tensor_option = torch::TensorOptions().dtype(at::kInt);
        while (task.pop_packet_from_input_queue(0, pkt))
        {
            if (pkt.timestamp() == bmf_sdk::Timestamp::BMF_EOF)
            {
                end_flag = true;
                BMFLOG(BMF_INFO) << "receive eof flags. final output " << m_output_idx << " frame.";
                task.set_timestamp(bmf_sdk::Timestamp::DONE);
                task.fill_output_packet(0, bmf_sdk::Packet::generate_eof_packet());
            }
            auto vframe = pkt.get<bmf_sdk::VideoFrame>();
            m_vframes.emplace_back(vframe);
            if (m_vframes.size() == m_numframes)
            {
                send(m_vframes, m_input_idx);
                m_vframes.clear();
                m_input_idx++;
                // vit;
                if (m_input_idx % m_batch_size == 0 || end_flag)
                {
                    // cudaSetDevice(m_gpu_id);
                    std::unique_lock<std::mutex> vit_lock(m_vit_mutex);
                    m_vit_cv.wait(vit_lock, [&]()
                                  { return (m_llm_input_queue.size() == m_batch_size) || (end_flag && m_output_idx == m_input_idx); });
                    int32_t batch_size = end_flag ? m_llm_input_queue.size() : m_batch_size;
                    std::vector<at::Tensor> new_image_features(batch_size);
                    std::vector<at::Tensor> input_ids(batch_size);
                    for (size_t i = 0; i < batch_size; i++)
                    {
                        auto tensor = m_llm_input_queue.front();
                        m_llm_input_queue.pop();
                        new_image_features[i] = tensor.cuda();
                        input_ids[i] = torch::cat({pre_ids,
                                                   torch::arange(m_vocab_size, m_vocab_size + tensor.sizes().data()[0], tensor_option),
                                                   post_ids});
                    }
                    kvcache_execute(new_image_features, input_ids);
                }
            }
            return 0;
        }
    }

private:
    void send(std::vector<bmf_sdk::VideoFrame> &vframes, int32_t m_input_idx)
    {
        auto task = std::make_unique<vit_task>(std::move(vframes), m_input_idx);
        task->set_dim(m_image_size, m_image_size, 15 * m_numframes, m_pool.get());
        m_vit->send(std::move(task));
    }

    void receive()
    {
        while (m_running)
        {
            if (m_output_idx < m_input_idx)
            {
                auto future = m_vit->receive();
                auto task = future.get();
                std::lock_guard<std::mutex> lock(m_vit_mutex);
                m_llm_input_queue.emplace(task->get_postprocess_output_tensor());
                m_output_idx++;
                if (m_llm_input_queue.size() == m_batch_size || end_flag)
                {
                    m_vit_cv.notify_one();
                }
            }
            else
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        }
    }

    void kvcache_execute(std::vector<at::Tensor> &new_image_features, std::vector<at::Tensor> &input_ids)
    {
        auto sample_config = tle::SamplingConfig();
        auto output_config = tle::OutputConfig();
        output_config.excludeInputFromOutput = true;
        auto size = new_image_features.size();
        if (m_executor->canEnqueueRequests())
        {
            for (size_t i = 0; i < size; i++)
            {
                tle::Shape shape(new_image_features[i].sizes().data(), new_image_features[i].sizes().size());
                tle::Tensor new_image_feature_tle = tle::Tensor::of((half *)new_image_features[i].data_ptr(), shape);
                tle::VecTokens cpu_inputids(input_ids[i].data_ptr<int32_t>(), input_ids[i].data_ptr<int32_t>() + input_ids[i].sizes().data()[0]);
                auto req = tle::Request(cpu_inputids, 256, false, sample_config, output_config,
                                        2, 32001, std::nullopt, std::nullopt, std::nullopt,
                                        std::nullopt,
                                        tle::PromptTuningConfig(new_image_feature_tle));
                auto req_id = m_executor->enqueueRequest(req);
            }
        }
        int32_t num_finished = 0;
        while (num_finished < size)
        {
            auto responses = m_executor->awaitResponses(std::chrono::milliseconds(10000));
            for (auto response : responses)
            {
                nlohmann::ordered_json json;
                num_finished++;
                auto req_id = response.getRequestId();
                json["req_id"] = req_id;
                json["prompt"] = cap_prompt;
                if (!response.hasError())
                {
                    auto result = response.getResult();
                    auto output_token_ids = result.outputTokenIds[0];
                    auto output_token = m_tokenizer->decode(output_token_ids);
                    BMFLOG(BMF_INFO) << output_token;
                    json["text"] = output_token;
                }
                json["answer_id"] = shortuuid();
                json["model_id"] = "llava";
                ordered_json max_json, metajson;
                max_json["model_max_length"] = 32768;
                json["params"] = max_json;
                json["metadata"] = metajson;
                std::lock_guard<std::mutex> lock(m_result_mutex);
                m_result_queue.emplace(json);
                m_result_cv.notify_one();
            }
        }
    }

    void write_results_to_file()
    {
        while (m_running)
        {
            std::unique_lock<std::mutex> lock(m_result_mutex);
            if (m_result_queue.empty())
            {
                lock.unlock();
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                continue;
            }
            auto result_json = m_result_queue.front();
            m_result_queue.pop();
            lock.unlock();
            ans_file << result_json << std::endl;
        }
    }

    std::string shortuuid()
    {
        boost::uuids::uuid uuid = boost::uuids::random_generator()();
        return boost::uuids::to_string(uuid);
    }

    bool m_running = false;
    volatile int32_t m_input_idx = 0;
    volatile int32_t m_output_idx = 0;
    std::vector<int32_t> m_gpu_ids{};
    int32_t m_numframes = 8;
    int32_t m_image_size = 336;
    int32_t m_patch_size = 14;
    int32_t m_num_patches = 24;
    int32_t m_downsample = 1;
    int32_t m_vocab_size = 32000;
    int32_t m_hidden_size = 4096;
    int32_t m_batch_size = 1;

    std::vector<int32_t> m_gpus{};
    std::vector<bmf_sdk::VideoFrame> m_vframes{};
    at::Tensor pre_ids;
    at::Tensor post_ids;
    at::Tensor prompt_ids;
    std::vector<std::vector<int32_t>> m_grid_pinpoints;
    std::unique_ptr<vit> m_vit = nullptr;
    std::unique_ptr<cpu_pool> m_pool = nullptr;
    std::queue<ordered_json> m_result_queue;
    std::queue<at::Tensor> m_llm_input_queue;
    // multi thread
    std::condition_variable m_vit_cv;
    std::mutex m_vit_mutex{};
    std::condition_variable m_result_cv;
    std::mutex m_result_mutex{};
    std::unique_ptr<tokenizer> m_tokenizer;
    // tensorrt llm
    std::unique_ptr<tle::Executor> m_executor = nullptr;
    // thread
    std::thread m_vit_thread{};
    std::thread m_result_thread{};
    std::ofstream ans_file;
    std::string result_path = "video_caption_result.json";
    std::string cap_prompt = "Describe this image in detail";

    bool end_flag = false;
};

std::shared_ptr<bmf_sdk::Module> constructor_caption(int node_id, bmf_sdk::JsonParam json_param)
{
    return std::make_shared<caption>(node_id, json_param);
}

static bmf_sdk::ModuleRegister r_constructor_caption("caption", BMF_SDK_VERSION, constructor_caption);
