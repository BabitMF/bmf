#include "QnnControlNetPipeLine.h"
#include <algorithm>

static void QnnLogCallBack(const char *fmt, QnnLog_Level_t level,
                           uint64_t timestamp, va_list argp) {
    switch (level) {
    case QNN_LOG_LEVEL_ERROR:
        BMFLITE_LOGE("controlnet", "timestamp: %lu, ", timestamp);
        BMFLITE_VLOGE("controlnet", fmt, argp);
        break;
    case QNN_LOG_LEVEL_WARN:
        BMFLITE_LOGW("controlnet", "timestamp: %lu, ", timestamp);
        BMFLITE_VLOGW("controlnet", fmt, argp);
        break;
    case QNN_LOG_LEVEL_INFO:
        BMFLITE_LOGI("controlnet", "timestamp: %lu, ", timestamp);
        BMFLITE_VLOGI("controlnet", fmt, argp);
        break;
    case QNN_LOG_LEVEL_VERBOSE:
        BMFLITE_LOGI("controlnet", "timestamp: %lu, ", timestamp);
        BMFLITE_VLOGI("controlnet", fmt, argp);
        break;
    case QNN_LOG_LEVEL_DEBUG:
        BMFLITE_LOGD("controlnet", "timestamp: %lu, ", timestamp);
        BMFLITE_VLOGD("controlnet", fmt, argp);
        break;
    case QNN_LOG_LEVEL_MAX:
        BMFLITE_LOGD("controlnet", "timestamp: %lu, ", timestamp);
        BMFLITE_VLOGD("controlnet", fmt, argp);
        break;
    default:
        BMFLITE_LOGD("controlnet", "timestamp: %lu, ", timestamp);
        BMFLITE_VLOGD("controlnet", fmt, argp);
        break;
    }
    return;
}
int QnnControlNetPipeline::init(const std::string htp_path,
                                const std::string system_path,
                                const std::string tokenizer_path,
                                const std::string unet_path,
                                const std::string text_encoder_path,
                                const std::string vae_path,
                                const std::string control_net_path) {
    if (htp_runtime_ == nullptr) {
        htp_runtime_ = std::make_shared<QnnHTPRuntime>();
    }
    if (htp_runtime_->init(htp_path, system_path, QNN_LOG_LEVEL_ERROR,
                           QnnLogCallBack) == false) {
        BMFLITE_LOGE("controlnet", "htp_runtime_ init error\n");
        return -1;
    }
    if (unet == nullptr) {
        unet = std::make_shared<QnnModel>();
    }
    if (unet->init(htp_runtime_, unet_path) == false) {
        BMFLITE_LOGE("controlnet", "unet init error\n");
        return -2;
    }
    if (text_encoder == nullptr) {
        text_encoder = std::make_shared<QnnModel>();
    }
    if (text_encoder->init(htp_runtime_, text_encoder_path) == false) {
        BMFLITE_LOGE("controlnet", "text_encoder inite error\n");
        return -3;
    }
    if (vae == nullptr) {
        vae = std::make_shared<QnnModel>();
    }
    if (vae->init(htp_runtime_, vae_path) == false) {
        BMFLITE_LOGE("controlnet", "vae inite error\n");
        return -4;
    }
    if (control_net == nullptr) {
        control_net = std::make_shared<QnnModel>();
    }
    if (control_net->init(htp_runtime_, control_net_path) == false) {
        BMFLITE_LOGE("controlnet", "control_net inite error\n");
        return -5;
    }

    if (tokenizer.load(tokenizer_path) < 0) {
        BMFLITE_LOGE("controlnet", "tokenizer load failed!\n");
        return -6;
    }

    scheduler.set_timesteps(max_step_);

    tokens = std::make_shared<QnnTensorData>(
        text_encoder->query_input_by_name("input_1"));
    text_latents = std::make_shared<QnnTensorData>(
        text_encoder->query_output_by_name("output_1"));
    bad_text_latents = std::make_shared<QnnTensorData>(
        text_encoder->query_output_by_name("output_1"));
    noise_latent =
        std::make_shared<QnnTensorData>(unet->query_input_by_name("input_1"));
    noise_latent_out =
        std::make_shared<QnnTensorData>(unet->query_output_by_name("output_1"));
    vae_in =
        std::make_shared<QnnTensorData>(vae->query_input_by_name("input_1"));
    time_embeddings = std::make_shared<QnnTensorData>(
        control_net->query_input_by_name("input_2"));
    canny_image = std::make_shared<QnnTensorData>(
        control_net->query_input_by_name("input_4"));
    out_image =
        std::make_shared<QnnTensorData>(vae->query_output_by_name("output_1"));

    auto controlnet_outs = control_net->get_all_output_names();
    for (auto &name : controlnet_outs) {
        controlnet_blks.push_back(std::make_shared<QnnTensorData>(
            control_net->query_output_by_name(name),
            unet->query_input_by_name(name)));
    }
    std::vector<uint8_t *> text_encoder_inputs{
        (uint8_t *)tokens->get_tensor().v1.clientBuf.data};
    std::vector<uint8_t *> text_encoder_outputs{
        (uint8_t *)text_latents->get_tensor().v1.clientBuf.data};
    text_encoder->register_inout_buffer(text_encoder_inputs,
                                        text_encoder_outputs);

    std::vector<uint8_t *> unet_inputs{
        (uint8_t *)noise_latent->get_tensor().v1.clientBuf.data,
        (uint8_t *)time_embeddings->get_tensor().v1.clientBuf.data,
        (uint8_t *)text_latents->get_tensor().v1.clientBuf.data};
    for (auto &blk : controlnet_blks) {
        unet_inputs.push_back((uint8_t *)blk->get_tensor().v1.clientBuf.data);
    }
    std::vector<uint8_t *> unet_outputs{
        (uint8_t *)noise_latent_out->get_tensor().v1.clientBuf.data};
    unet->register_inout_buffer(unet_inputs, unet_outputs);

    std::vector<uint8_t *> controlnet_inputs{
        (uint8_t *)noise_latent->get_tensor().v1.clientBuf.data,
        (uint8_t *)time_embeddings->get_tensor().v1.clientBuf.data,
        (uint8_t *)text_latents->get_tensor().v1.clientBuf.data,
        (uint8_t *)canny_image->get_tensor().v1.clientBuf.data};

    std::vector<uint8_t *> controlnet_outputs{};
    for (auto &blk : controlnet_blks) {
        controlnet_outputs.push_back(
            (uint8_t *)blk->get_tensor().v1.clientBuf.data);
    }

    control_net->register_inout_buffer(controlnet_inputs, controlnet_outputs);

    std::vector<uint8_t *> vae_inputs{
        (uint8_t *)vae_in->get_tensor().v1.clientBuf.data};
    std::vector<uint8_t *> vae_outputs{
        (uint8_t *)out_image->get_tensor().v1.clientBuf.data};
    vae->register_inout_buffer(vae_inputs, vae_outputs);

    inited_ = true;
    return 0;
}

int QnnControlNetPipeline::tokenize(std::string positive_prompt_en,
                                    std::string negative_prompt_en,
                                    float *canny_ptr, int max_step, int seed) {
    if (inited_ == false) {
        return -1;
    }

    canny_image->from_float(canny_ptr);

    auto tokens_and_weights = tokenizer.tokenize(positive_prompt_en, 77, true);
    auto bad_tokens_and_weights =
        tokenizer.tokenize(negative_prompt_en, 77, true);

    auto tokenized = tokens_and_weights.first;
    auto bad_tokenized = bad_tokens_and_weights.first;

    tokens->from_int(tokenized.data());
    text_encoder->register_output_buffer(
        0, (uint8_t *)text_latents->get_tensor().v1.clientBuf.data);
    if (!text_encoder->inference()) {
        BMFLITE_LOGE("controlnet", "text_encoder inference failed");
        return -7;
    }

    tokens->from_int(bad_tokenized.data());
    text_encoder->register_output_buffer(
        0, (uint8_t *)bad_text_latents->get_tensor().v1.clientBuf.data);
    if (!text_encoder->inference()) {
        BMFLITE_LOGE("controlnet", "text_encoder inference failed");
        return -7;
    }

    if (max_step != max_step_) {
        if (max_step != 50 || max_step != 20) {
            BMFLITE_LOGW("controlnet", "step not valid");
            max_step_ = 20;
        } else {
            max_step_ = max_step;
        }
        scheduler.set_timesteps(max_step_);
    }
    current_step_ = 0;
    latent_input_ = scheduler.randn_mat(seed, 64 * 64 * 4, 1);

    return 0;
}

int QnnControlNetPipeline::step(float *image_out, int step) {
    if (inited_ == false) {
        return -1;
    }

    if (current_step_ >= max_step_) {
        BMFLITE_LOGW("controlnet", "step out of range"); // Generated complete
        return 2;
    }

    int ongoing_step = std::min(current_step_ + step, max_step_);
    std::vector<float> denoised(64 * 64 * 4);
    std::vector<float> bad_denoised(64 * 64 * 4);

    for (; current_step_ < ongoing_step; current_step_++) {
        time_embeddings->from_float(
            time_embedding_input_map[max_step_][current_step_]);
        noise_latent->from_float((float *)latent_input_.data());
        control_net->register_input_buffer(
            2, (uint8_t *)text_latents->get_tensor().v1.clientBuf.data);
        if (!control_net->inference()) {
            BMFLITE_LOGE("controlnet", "control_net inference failed");
            return -8;
        }
        for (auto &blk : controlnet_blks) {
            blk->in_place_adjust_quantization();
        }
        unet->register_input_buffer(
            2, (uint8_t *)text_latents->get_tensor().v1.clientBuf.data);
        if (!unet->inference()) {
            BMFLITE_LOGE("controlnet", "unet inference failed");
            return -9;
        }
        noise_latent_out->to_float((float *)denoised.data());

        control_net->register_input_buffer(
            2, (uint8_t *)bad_text_latents->get_tensor().v1.clientBuf.data);
        if (!control_net->inference()) {
            BMFLITE_LOGE("controlnet", "control_net inference failed");
            return -8;
        }
        for (auto &blk : controlnet_blks) {
            blk->in_place_adjust_quantization();
        }
        unet->register_input_buffer(
            2, (uint8_t *)bad_text_latents->get_tensor().v1.clientBuf.data);
        if (!unet->inference()) {
            BMFLITE_LOGE("controlnet", "unet inference failed");
            return -9;
        }
        noise_latent_out->to_float((float *)bad_denoised.data());

        for (int j = 0; j < denoised.size(); j++) {
            denoised[j] =
                7.5 * (denoised[j] - bad_denoised[j]) + bad_denoised[j];
        }

        latent_input_ = scheduler.step(current_step_, latent_input_, denoised);
    }
    vae_in->from_float((float *)latent_input_.data());

    if (!vae->inference()) {
        BMFLITE_LOGE("controlnet", "vae inference failed");
        return -10;
    }

    out_image->to_float((float *)image_out);

    return 0;
}