#ifndef STABLEDIFFUSION_SCHEDULER_DPMPP_2M_H
#define STABLEDIFFUSION_SCHEDULER_DPMPP_2M_H

#include "scheduler_base.h"
#include <vector>
#include <array>

class scheduler_dpmpp_2m : public scheduler_base {
  public:
    scheduler_dpmpp_2m();

    std::vector<float>
    dpm_solver_first_order_update(std::vector<float> model_output, int timestep,
                                  int prev_timestep, std::vector<float> sample,
                                  std::vector<float> noise);
    std::vector<float> multistep_dpm_solver_second_order_update(
        std::array<std::vector<float>, 2> model_output_list,
        std::array<int, 2> timestep_list, int prev_timestep,
        std::vector<float> sample, std::vector<float> noise);
    virtual std::vector<float> set_timesteps(int steps) override;

    virtual std::vector<float> scale_model_input(std::vector<float> &sample,
                                                 int step_index) override;

    virtual std::vector<float> step(int step_index,
                                    std::vector<float> &sample_mat,
                                    std::vector<float> &denoised) override;

    virtual std::vector<float> randn_mat(int seed, int size,
                                         int is_latent_sample) override;

    virtual float getInitNoiseSigma() override;

    virtual std::vector<float> get_timesteps() override;

    virtual std::vector<float> get_sigmas() override;

    void set_init_sigma(float sigma) override;

  private:
    float beta_start = 0.00085f;
    float beta_end = 0.012f;
    std::string beta_schedule = "scaled_linear";
    std::vector<float> trained_betas;
    const int solver_order = 2;
    bool thresholding = false;
    float dynamic_thresholding_ratio = 0.995f;
    float sample_max_value = 1.0f;
    std::string algorithm_type = "dpmsolver++";
    std::string solver_type = "midpoint";
    bool lower_order_final = true;
    bool clip_sample = false;
    float clip_sample_range = 1.0f;
    std::array<std::vector<float>, 2> model_outputs = {};
    int num_train_timesteps = 1000;
    std::vector<float> alphas, betas, alphas_cumprod, timesteps, sigmas,
        sigmas_total, alpha_ts, sigma_ts, lambda_ts;
    int rand_seed;
    int num_inference_steps = 8;
    float init_noise_sigma = 1.0f;
};

#endif // STABLEDIFFUSION_SCHEDULER_DPMPP_2M_H
