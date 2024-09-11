#ifndef STABLEDIFFUSION_SCHEDULER_BASE_H
#define STABLEDIFFUSION_SCHEDULER_BASE_H

#include <vector>
#include <string>

class scheduler_base {
  public:
    virtual std::vector<float> set_timesteps(int num_inference_steps) = 0;

    virtual std::vector<float> scale_model_input(std::vector<float> &sample,
                                                 int step_index) = 0;

    virtual std::vector<float> step(int step_index,
                                    std::vector<float> &sample_mat,
                                    std::vector<float> &denoised) = 0;

    virtual float getInitNoiseSigma() = 0;

    virtual std::vector<float> randn_mat(int seed, int size,
                                         int is_latent_sample) = 0;

    virtual std::vector<float> get_timesteps() = 0;

    virtual std::vector<float> get_sigmas() = 0;

    virtual void set_init_sigma(float sigma) = 0;
};

#endif // STABLEDIFFUSION_SCHEDULER_BASE_H
