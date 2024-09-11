#include <iostream>
#include <memory>
#include <string>

#include "inference/QnnModel.h"
#include "scheduler/scheduler_dpmpp_2m.h"
#include "utils/time_all.h"
#include "utils/utils_tokenizer.h"

class QnnControlNetPipeline {
  public:
    QnnControlNetPipeline() = default;
    ~QnnControlNetPipeline() = default;
    int
    init(const std::string htp_path, const std::string system_path,
         const std::string tokenizer_path = "/data/local/tmp/ControlNetData",
         const std::string unet_path =
             "/data/local/tmp/ControlNetData/unet.serialized.bin",
         const std::string text_encoder_path =
             "/data/local/tmp/ControlNetData/text_encoder.serialized.bin",
         const std::string vae_path =
             "/data/local/tmp/ControlNetData/vae_decoder.serialized.bin",
         const std::string control_net_path =
             "/data/local/tmp/ControlNetData/controlnet.serialized.bin");
    int tokenize(std::string positive_prompt_en, std::string negative_prompt_en,
                 float *canny_ptr, int max_step = 50, int seed = 0);
    int step(float *image_out, int step);

  private:
    std::shared_ptr<QnnHTPRuntime> htp_runtime_;

    std::shared_ptr<QnnModel> unet, text_encoder, vae, control_net;

    std::shared_ptr<QnnTensorData> tokens, text_latents, bad_text_latents,
        noise_latent, time_embeddings, canny_image, noise_latent_out, vae_in,
        out_image;

    std::vector<std::shared_ptr<QnnTensorData>> controlnet_blks;

    scheduler_dpmpp_2m scheduler;

    CLIPTokenizer tokenizer;
    std::vector<float> latent_input_;
    int max_step_ = 50;
    int current_step_ = 0;
    bool inited_ = false;
};