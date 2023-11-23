/*
 * Copyright 2023 Babit Authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "maxine_module.h"
#include "bmf/sdk/log.h"
#include "hmp/imgproc.h"
#include <stdexcept>

#define CHECK_MAXINE(func, msg)                                                        \
    do {                                                                               \
        auto status = func;                                                            \
        if (status != NVCV_SUCCESS) {                                                  \
            BMFLOG_NODE(BMF_ERROR, node_id_) << msg <<" at line " << __LINE__;         \
            return status;                                                             \
        }                                                                              \
    } while (0);                                                                       \

MaxineModule::MaxineModule(int node_id, JsonParam option)
    : Module(node_id, option) {
    stream_ = 0;

    // common params
    option.get_int("num_effects", num_effects_);
    vfx_effects_.resize(num_effects_);

    if (option.has_key("model_dir")) {
        option.get_string("model_dir", model_dir_);
    }

    width_ = 0;
    height_ = 0;
    resolution_ = 0;
    max_num_streams_ = 1;
    comp_mode_ = compWhite;
    bg_color_ = nullptr;
    batch_of_states_ = nullptr;
    has_allocated_ = false;

    option_ = option;

    int err = initialize();

    if (err != NVCV_SUCCESS) {
        throw std::runtime_error("Initialize failed");
    }
}

int MaxineModule::initialize() {
    /**
     * ArtifactReduction
     * CudaStream (CUstream)     - CUDA stream
     * SrcImage0  (NvCVImage*)   - input  image normalized GPU BGR_F32 planar [Between 90p and 1080p; Preferred width and height divisble by 8 - this reduces extra resize/pad GPU kernels. ]
     * DstImage0  (NvCVImage*)   - output image normalized GPU BGR_F32 planar [same as input]
     * ModelDir   (const char*)  - path of directory containing the model
     * Mode   (unsigned int) - mode of the filter [0 -conservative or 1 - aggressive]
     * BatchSize  (unsigned int) - batch size of the images (default 1)
     * ModelBatch (unsigned int) - batch size of the model (default 1; choices 1)
    **/
    if (option_.has_key("ArtifactReduction")) {
        JsonParam effect_params;
        option_.get_object("ArtifactReduction", effect_params);
        int index;
        effect_params.get_int("index", index);

        vfx_effects_[index].name_ = "ArtifactReduction";
        vfx_effects_[index].input_in_rgb_range_ = false;

        CHECK_MAXINE(NvVFX_CreateEffect("ArtifactReduction", &vfx_effects_[index].handle_), "Error creating ArtifactReduction effect");

        CHECK_MAXINE(NvVFX_SetString(vfx_effects_[index].handle_, NVVFX_MODEL_DIRECTORY, model_dir_.c_str()), "Error setting model path");
        
        if (effect_params.has_key("mode")) {
            int mode;
            effect_params.get_int("mode", mode);
            CHECK_MAXINE(NvVFX_SetU32(vfx_effects_[index].handle_, NVVFX_MODE, mode), "Error setting mode");
        }

        if (effect_params.has_key("batch_size")) {
            int batch_size;
            effect_params.get_int("batch_size", batch_size);
            CHECK_MAXINE(NvVFX_SetU32(vfx_effects_[index].handle_, NVVFX_BATCH_SIZE, batch_size), "Error setting batch size");
        }

        if (effect_params.has_key("model_batch")) {
            int model_batch;
            effect_params.get_int("model_batch", model_batch);
            CHECK_MAXINE(NvVFX_SetU32(vfx_effects_[index].handle_, NVVFX_MODEL_BATCH, model_batch), "Error setting model batch");
        }

        vfx_effects_[index].gpu_buf_src_.reserve(1);
        vfx_effects_[index].gpu_buf_dst_.reserve(1);
    }

	/**
     * SuperRes
     * CudaStream (CUstream)     - CUDA stream
     * SrcImage0  (NvCVImage*)   - input  image normalized GPU BGR_F32 planar ([90p, 1080p] for up to 2x, [90, 720p] for 3x, [90p, 540p] for 4x. ; Preferred width and height divisible by 2 - this reduces extra resize/pad GPU kernels.)
     * DstImage0  (NvCVImage*)   - output image normalized GPU BGR_F32 planar (Input x Scale)
     * ModelDir   (const char*)  - path of directory containing the model
     * Mode       (unsigned int) - mode of the filter [0 -conservative or 1 -aggressive]
     * BatchSize  (unsigned int) - batch size of the images (default 1)
     * ModelBatch (unsigned int) - batch size of the model (default 1; choices 1)
     * Strength   (float)        - the strength of the sharpness, typically [0, 4.0] (default 1.0) 
    **/
    if (option_.has_key("SuperRes")) {
        JsonParam effect_params;
        option_.get_object("SuperRes", effect_params);
        int index;
        effect_params.get_int("index", index);

        vfx_effects_[index].name_ = "SuperRes";
        vfx_effects_[index].input_in_rgb_range_ = false;

        CHECK_MAXINE(NvVFX_CreateEffect("SuperRes", &vfx_effects_[index].handle_), "Error creating SuperRes effect");

        CHECK_MAXINE(NvVFX_SetString(vfx_effects_[index].handle_, NVVFX_MODEL_DIRECTORY, model_dir_.c_str()), "Error setting model path");

        if (effect_params.has_key("mode")) {
            int mode;
            effect_params.get_int("mode", mode);
            CHECK_MAXINE(NvVFX_SetU32(vfx_effects_[index].handle_, NVVFX_MODE, mode), "Error setting mode");
        }

        if (effect_params.has_key("batch_size")) {
            int batch_size;
            effect_params.get_int("batch_size", batch_size);
            CHECK_MAXINE(NvVFX_SetU32(vfx_effects_[index].handle_, NVVFX_BATCH_SIZE, batch_size), "Error setting batch size");
        }

        if (effect_params.has_key("model_batch")) {
            int model_batch;
            effect_params.get_int("model_batch", model_batch);
            CHECK_MAXINE(NvVFX_SetU32(vfx_effects_[index].handle_, NVVFX_MODEL_BATCH, model_batch), "Error setting model batch");
        }

        if (effect_params.has_key("strength")) {
            double tmp;
            effect_params.get_double("strength", tmp);
            float strength = (float) tmp;
            CHECK_MAXINE(NvVFX_SetU32(vfx_effects_[index].handle_, NVVFX_STRENGTH, strength), "Error setting strength");
        }

        effect_params.get_int("resolution", resolution_);
        vfx_effects_[index].gpu_buf_src_.reserve(1);
        vfx_effects_[index].gpu_buf_dst_.reserve(1);
    }

    /**
     * Upscale
     * CudaStream (CUstream)     - CUDA stream
     * SrcImage0  (NvCVImage*)   - input  image buffer in GPU RGBA U8 chunky format, pitch alignment=32
     * DstImage0  (NvCVImage*)   - output image buffer in GPU RGBA U8 chunky format with supported scaling of 1.33x, 1.5x, 2x, 3x or 4x w.r.t. input resolution
     * Strength   (float)        - strength of the filter from 0 to 1 (default 0.4)
     * BatchSize  (unsigned int) - batch size (default 1)
     * ModelBatch (unsigned int) - batch size of the model (default 1; choices 1)
    **/
    if (option_.has_key("Upscale")) {
        JsonParam effect_params;
        option_.get_object("Upscale", effect_params);
        int index;
        effect_params.get_int("index", index);

        vfx_effects_[index].name_ = "Upscale";
        vfx_effects_[index].input_in_rgb_range_ = true;

        CHECK_MAXINE(NvVFX_CreateEffect("Upscale", &vfx_effects_[index].handle_), "Error creating Upscale effect");

        if (effect_params.has_key("batch_size")) {
            int batch_size;
            effect_params.get_int("batch_size", batch_size);
            CHECK_MAXINE(NvVFX_SetU32(vfx_effects_[index].handle_, NVVFX_BATCH_SIZE, batch_size), "Error setting batch size");
        }

        if (effect_params.has_key("model_batch")) {
            int model_batch;
            effect_params.get_int("model_batch", model_batch);
            CHECK_MAXINE(NvVFX_SetU32(vfx_effects_[index].handle_, NVVFX_MODEL_BATCH, model_batch), "Error setting model batch");
        }

        if (effect_params.has_key("strength")) {
            double tmp;
            effect_params.get_double("strength", tmp);
            float strength = (float) tmp;
            CHECK_MAXINE(NvVFX_SetU32(vfx_effects_[index].handle_, NVVFX_STRENGTH, strength), "Error setting strength");
        }

        effect_params.get_int("resolution", resolution_);
        vfx_effects_[index].gpu_buf_src_.reserve(1);
        vfx_effects_[index].gpu_buf_dst_.reserve(1);
    }

	/**
     * GreenScreen
     * CudaGraph (unsigned int) - enable cuda graph (0|1)CudaStream (CUstream)     - CUDA stream
     * SrcImage0  (NvCVImage*)   - input  image buffer [Format req. - BGR_U8 GPU (chunky); Preferred aspect ratio - 16:9, this reduces extra resize/pad GPU kernels.]
     * DstImage0  (NvCVImage*)   - output image buffer [Format req. -   A_U8 GPU         ; Size req. - (equal to the input image size);            ]
     * ModelDir   (const char*)  - path of directory containing the model
     * BatchSize  (unsigned int) - batch size of the images (default 1)
     * ModelBatch (unsigned int) - batch size of the model (default 1; choices 1 & 8)
     * Mode       (unsigned int) - AIGS mode available
           0 - Best quality
           1 - Best performance
     * MaxInputWidth (unsgined int) - maximum width of the input tensor  (default 3840)
     * MaxInputHeight (unsgiend int) - maximum height of the input tensor (default 2160)
     * State           (void*)        - address of an array of state variables
     * MaxNumberStreams (unsigned int) - maximum number of concurrent input streams (default 1)
    **/
    if (option_.has_key("GreenScreen")) {
        JsonParam effect_params;
        option_.get_object("GreenScreen", effect_params);
        int index;
        effect_params.get_int("index", index);

        vfx_effects_[index].name_ = "GreenScreen";
        vfx_effects_[index].input_in_rgb_range_ = true;

        CHECK_MAXINE(NvVFX_CreateEffect("GreenScreen", &vfx_effects_[index].handle_), "Error creating GreenScreen effect");

        CHECK_MAXINE(NvVFX_SetString(vfx_effects_[index].handle_, NVVFX_MODEL_DIRECTORY, model_dir_.c_str()), "Error setting model path");

        if (effect_params.has_key("mode")) {
            int mode;
            effect_params.get_int("mode", mode);
            CHECK_MAXINE(NvVFX_SetU32(vfx_effects_[index].handle_, NVVFX_MODE, mode), "Error setting mode");
        }

        if (effect_params.has_key("batch_size")) {
            int batch_size;
            effect_params.get_int("batch_size", batch_size);
            CHECK_MAXINE(NvVFX_SetU32(vfx_effects_[index].handle_, NVVFX_BATCH_SIZE, batch_size), "Error setting batch size");
        }

        if (effect_params.has_key("model_batch")) {
            int model_batch;
            effect_params.get_int("model_batch", model_batch);
            CHECK_MAXINE(NvVFX_SetU32(vfx_effects_[index].handle_, NVVFX_MODEL_BATCH, model_batch), "Error setting model batch");
        }

        if (effect_params.has_key("max_input_width")) {
            int max_input_width;
            effect_params.get_int("max_input_width", max_input_width);
            CHECK_MAXINE(NvVFX_SetU32(vfx_effects_[index].handle_, NVVFX_MAX_INPUT_WIDTH, max_input_width), "Error setting max input width");
        }

        if (effect_params.has_key("max_input_height")) {
            int max_input_height;
            effect_params.get_int("max_input_height", max_input_height);
            CHECK_MAXINE(NvVFX_SetU32(vfx_effects_[index].handle_, NVVFX_MAX_INPUT_HEIGHT, max_input_height), "Error setting max input height");
        }

        if (effect_params.has_key("max_num_streams")) {
            effect_params.get_int("max_num_streams", max_num_streams_);
            CHECK_MAXINE(NvVFX_SetU32(vfx_effects_[index].handle_, NVVFX_MAX_NUMBER_STREAMS, max_num_streams_), "Error setting max input height");
        }

		int tmp_mode;
        effect_params.get_int("comp_mode", tmp_mode);
        comp_mode_ = CompMode(tmp_mode);
        if (comp_mode_ == compBG) {
            std::string bg_file;
            effect_params.get_string("bg_file", bg_file);
            bg_img_ = cv::imread(bg_file.c_str());
        }

        vfx_effects_[index].gpu_buf_src_.reserve(1);
        vfx_effects_[index].gpu_buf_dst_.reserve(2);
    }

	/**
     * BackgroundBlur
     * CudaStream  (CUstream)     - CUDA stream (default 0)
     * SrcImage0  (NvCVImage*)   - input BGR-U8 image buffer
     * SrcImage1  (NvCVImage*)   - input Y or A-U8 mask buffer
     * DstImage0  (NvCVImage*)   - output image buffer
     * Strength  (float)        - strength - amount of blur (default 0.5f)
     * BatchSize  (unsigned int) - batch size of the images (default 1)
     * ModelBatch  (unsigned int) - batch size of the model (default 1; the only supported model batch is 1)
    **/
    if (option_.has_key("BackgroundBlur")) {
        JsonParam effect_params;
        option_.get_object("BackgroundBlur", effect_params);
        int index;
        effect_params.get_int("index", index);

        vfx_effects_[index].name_ = "BackgroundBlur";
        vfx_effects_[index].input_in_rgb_range_ = true;

        if (index < 1 || vfx_effects_[index - 1].name_ != "GreenScreen") {
            BMFLOG_NODE(BMF_ERROR, node_id_) << "BackgroundBlur must be behind GreenScreen";
            return -1;
        }

        CHECK_MAXINE(NvVFX_CreateEffect("BackgroundBlur", &vfx_effects_[index].handle_), "Error creating BackgroundBlur effect");

        if (effect_params.has_key("batch_size")) {
            int batch_size;
            effect_params.get_int("batch_size", batch_size);
            CHECK_MAXINE(NvVFX_SetU32(vfx_effects_[index].handle_, NVVFX_BATCH_SIZE, batch_size), "Error setting batch size");
        }

        if (effect_params.has_key("model_batch")) {
            int model_batch;
            effect_params.get_int("model_batch", model_batch);
            CHECK_MAXINE(NvVFX_SetU32(vfx_effects_[index].handle_, NVVFX_MODEL_BATCH, model_batch), "Error setting model batch");
        }

        if (effect_params.has_key("strength")) {
            double tmp;
            effect_params.get_double("strength", tmp);
            float strength = (float) tmp;
            CHECK_MAXINE(NvVFX_SetU32(vfx_effects_[index].handle_, NVVFX_STRENGTH, strength), "Error setting strength");
        }

        vfx_effects_[index].gpu_buf_dst_.reserve(1);
    }

	/**
     * Denoising
     * CudaStream      (CUstream)     - CUDA stream
     * SrcImage0       (NvCVImage*)   - input  image normalized GPU BGR_F32 planar
     * DstImage0       (NvCVImage*)   - output image normalized GPU BGR_F32 planar [same as input]
     * ModelDir        (const char*)  - path of directory containing the model
     * BatchSize       (unsigned int) - batch size of the images (default 1)
     * ModelBatch      (unsigned int) - batch size of the model (default 1; choices 1)
     * Strength        (float)        - strength of denoising [0-1]
     * StrengthLevels  (unsigned int) - number of strength levels (how many unique strength values in the interval [0,1])
     * State           (void*)        - address of an array of state variables
     * StateSize       (unsigned int) - size of one state variable in bytes
    **/
    if (option_.has_key("Denoising")) {
        JsonParam effect_params;
        option_.get_object("Denoising", effect_params);
        int index;
        effect_params.get_int("index", index);

        vfx_effects_[index].name_ = "Denoising";
        vfx_effects_[index].input_in_rgb_range_ = false;

        CHECK_MAXINE(NvVFX_CreateEffect("Denoising", &vfx_effects_[index].handle_), "Error creating Denoising effect");

        CHECK_MAXINE(NvVFX_SetString(vfx_effects_[index].handle_, NVVFX_MODEL_DIRECTORY, model_dir_.c_str()), "Error setting model path");

        if (effect_params.has_key("batch_size")) {
            int batch_size;
            effect_params.get_int("batch_size", batch_size);
            CHECK_MAXINE(NvVFX_SetU32(vfx_effects_[index].handle_, NVVFX_BATCH_SIZE, batch_size), "Error setting batch size");
        }

        if (effect_params.has_key("model_batch")) {
            int model_batch;
            effect_params.get_int("model_batch", model_batch);
            CHECK_MAXINE(NvVFX_SetU32(vfx_effects_[index].handle_, NVVFX_MODEL_BATCH, model_batch), "Error setting model batch");
        }

        if (effect_params.has_key("strength")) {
            double tmp;
            effect_params.get_double("strength", tmp);
            float strength = (float) tmp;
            CHECK_MAXINE(NvVFX_SetU32(vfx_effects_[index].handle_, NVVFX_STRENGTH, strength), "Error setting strength");
        }

        if (effect_params.has_key("strength_levels")) {
            int strength_levels;
            effect_params.get_int("strength_levels", strength_levels);
            CHECK_MAXINE(NvVFX_SetU32(vfx_effects_[index].handle_, NVVFX_STRENGTH_LEVELS, strength_levels), "Error setting strength level");
        }

        if (effect_params.has_key("state_size")) {
            int state_size;
            effect_params.get_int("state_size", state_size);
            CHECK_MAXINE(NvVFX_SetU32(vfx_effects_[index].handle_, NVVFX_STATE_SIZE, state_size), "Error setting state size");
        }

        vfx_effects_[index].gpu_buf_src_.reserve(1);
        vfx_effects_[index].gpu_buf_dst_.reserve(1);
    }
    return 0;
}

MaxineModule::~MaxineModule()
{
    if (!state_array_.empty()) {
        for (int i = 0; i < state_array_.size(); i++) {
            NvVFX_DeallocateState(vfx_effects_[0].handle_, state_array_[i]);
        }
        state_array_.clear();
    }
    if (batch_of_states_ != nullptr) {
        free(batch_of_states_);
        batch_of_states_ = nullptr;
    }
    for (int i = 0; i < num_effects_; i++) {
        NvVFX_DestroyEffect(vfx_effects_[i].handle_);
    }
}

NvCV_Status MaxineModule::check_scale_isotropy(const NvCVImage *src, const NvCVImage *dst)
{
    if (src->width * dst->height != src->height * dst->width) {
        printf("%ux%u --> %ux%u: different scale for width and height is not supported\n",
               src->width, src->height, dst->width, dst->height);
        return NVCV_ERR_RESOLUTION;
    }
    return NVCV_SUCCESS;
}

int MaxineModule::set_bg_color(CompMode mode)
{
    switch (mode) {
        case compGreen: {
            unsigned char h_bg_color[3] = {128, 255, 128};
            cudaMalloc(&bg_color_, sizeof(unsigned char) * 3);
            cudaMemcpy(bg_color_, h_bg_color, sizeof(unsigned char) * 3, cudaMemcpyHostToDevice);
        } break;
        case compWhite: {
            unsigned char h_bg_color[3] = {255, 255, 255};
            cudaMalloc(&bg_color_, sizeof(unsigned char) * 3);
            cudaMemcpy(bg_color_, h_bg_color, sizeof(unsigned char) * 3, cudaMemcpyHostToDevice);
        } break;
        case compBG: {
            float scale = float(height_) / float(bg_img_.rows);
            if ((scale * bg_img_.cols) < float(width_)) {
                scale = float(width_) / float(bg_img_.cols);
            }

            cv::Mat resized_bg;
            cv::resize(bg_img_, resized_bg, cv::Size(), scale, scale, cv::INTER_AREA);

            cv::Rect rect(0, 0, width_, height_);
            resized_cropped_bg_img_ = resized_bg(rect);
            NvCVImage bg_img_cpu;
            (void)NVWrapperForCVMat(&resized_cropped_bg_img_, &bg_img_cpu);
            CHECK_MAXINE(NvCVImage_Alloc(&vfx_bg_, width_, height_, NVCV_BGR, NVCV_U8, NVCV_INTERLEAVED, NVCV_GPU, 1), "Error allocating NvCVImage");
            CHECK_MAXINE(NvCVImage_Transfer(&bg_img_cpu, &vfx_bg_, 1.0f, stream_, &vfx_tmp_), "Error transferring image");
        } break;
    }
    return 0;
}

NvCV_Status MaxineModule::vfx_setup(VideoFrame& vframe) {
    NvCV_Status vfx_status = NVCV_SUCCESS;
    
    if (!has_allocated_ || (vframe.width() != width_ || vframe.height() != height_)) {
        width_ = vframe.width();
        height_ = vframe.height();
        if (has_allocated_) {
            has_allocated_ = false;
        }
    }
    // wrapper input frame
    Frame frm(vframe.frame());
    CHECK_MAXINE(NvCVImage_Init(&vfx_src_, width_, height_, width_ * 3, frm.plane_data(0),
                   NVCV_RGB, NVCV_U8, NVCV_INTERLEAVED, NVCV_GPU), "Error init wrapper for input frames");

    if (has_allocated_) {
        return NVCV_SUCCESS;
    }

    int dst_width = width_, dst_height = height_;

    for (int i = 0; i < num_effects_; i++) {
        auto &effect = vfx_effects_[i];
        std::string effect_name = effect.name_;
        if (!strcmp(effect_name.c_str(), NVVFX_FX_ARTIFACT_REDUCTION)) {
            CHECK_MAXINE(NvCVImage_Alloc(&effect.gpu_buf_src_[0], width_, height_, NVCV_BGR, NVCV_F32, NVCV_PLANAR, NVCV_GPU, 1), "Error allocating NvCVImage");
            CHECK_MAXINE(NvCVImage_Alloc(&effect.gpu_buf_dst_[0], width_, height_, NVCV_BGR, NVCV_F32, NVCV_PLANAR, NVCV_GPU, 1), "Error allocating NvCVImage");
            CHECK_MAXINE(NvVFX_SetImage(effect.handle_, NVVFX_INPUT_IMAGE, &effect.gpu_buf_src_[0]), "Error setting src image");
            CHECK_MAXINE(NvVFX_SetImage(effect.handle_, NVVFX_OUTPUT_IMAGE, &effect.gpu_buf_dst_[0]), "Error setting dst image");
            CHECK_MAXINE(NvVFX_SetCudaStream(effect.handle_, NVVFX_CUDA_STREAM, stream_), "Error setting cuda stream");
            CHECK_MAXINE(NvVFX_Load(effect.handle_), "Error Loading");
        }
        else if (!strcmp(effect_name.c_str(), NVVFX_FX_SUPER_RES)) {
            dst_width = width_ * resolution_ / height_;
            dst_height = resolution_;
            CHECK_MAXINE(NvCVImage_Alloc(&effect.gpu_buf_src_[0], width_, height_, NVCV_BGR, NVCV_F32, NVCV_PLANAR, NVCV_GPU, 1), "Error allocating NvCVImage");
            CHECK_MAXINE(NvCVImage_Alloc(&effect.gpu_buf_dst_[0], dst_width, resolution_, NVCV_BGR, NVCV_F32, NVCV_PLANAR, NVCV_GPU, 1), "Error allocating NvCVImage");
            CHECK_MAXINE(check_scale_isotropy(&effect.gpu_buf_src_[0], &effect.gpu_buf_dst_[0]), "Error checking scale isotropy");
            CHECK_MAXINE(NvVFX_SetImage(effect.handle_, NVVFX_INPUT_IMAGE, &effect.gpu_buf_src_[0]), "Error setting src image");
            CHECK_MAXINE(NvVFX_SetImage(effect.handle_, NVVFX_OUTPUT_IMAGE, &effect.gpu_buf_dst_[0]), "Error setting dst image");
            CHECK_MAXINE(NvVFX_SetCudaStream(effect.handle_, NVVFX_CUDA_STREAM, stream_), "Error setting cuda stream");
            CHECK_MAXINE(NvVFX_Load(effect.handle_), "Error Loading");
        }
        else if (!strcmp(effect_name.c_str(), NVVFX_FX_SR_UPSCALE)) {
            dst_width = width_ * resolution_ / height_;
            dst_height = resolution_;
            CHECK_MAXINE(NvCVImage_Alloc(&effect.gpu_buf_src_[0], width_, height_, NVCV_RGBA, NVCV_U8, NVCV_INTERLEAVED, NVCV_GPU, 32), "Error allocating NvCVImage");
            CHECK_MAXINE(NvCVImage_Alloc(&effect.gpu_buf_dst_[0], dst_width, resolution_, NVCV_RGBA, NVCV_U8, NVCV_INTERLEAVED, NVCV_GPU, 32), "Error allocating NvCVImage");
            CHECK_MAXINE(check_scale_isotropy(&effect.gpu_buf_src_[0], &effect.gpu_buf_dst_[0]), "Error checking scale isotropy");
            CHECK_MAXINE(NvVFX_SetImage(effect.handle_, NVVFX_INPUT_IMAGE, &effect.gpu_buf_src_[0]), "Error setting src image");
            CHECK_MAXINE(NvVFX_SetImage(effect.handle_, NVVFX_OUTPUT_IMAGE, &effect.gpu_buf_dst_[0]), "Error setting dst image");
            CHECK_MAXINE(NvVFX_SetCudaStream(effect.handle_, NVVFX_CUDA_STREAM, stream_), "Error setting cuda stream");
            CHECK_MAXINE(NvVFX_Load(effect.handle_), "Error Loading");
        }
        else if (!strcmp(effect_name.c_str(), NVVFX_FX_GREEN_SCREEN)) {
            if (i != num_effects_ - 1 && strcmp(vfx_effects_[i+1].name_.c_str(), NVVFX_FX_BGBLUR)) {
                BMFLOG_NODE(BMF_ERROR, node_id_) << "GreenScreen effect should be the last or the second last followed by a Background Blur effect currently";
                return NVCV_ERR_GENERAL;
            }
            CHECK_MAXINE(NvCVImage_Alloc(&effect.gpu_buf_src_[0], width_, height_, NVCV_BGR, NVCV_U8, NVCV_CHUNKY, NVCV_GPU, 1), "Error allocating NvCVImage");
            CHECK_MAXINE(NvCVImage_Alloc(&effect.gpu_buf_dst_[0], width_, height_, NVCV_A, NVCV_U8, NVCV_CHUNKY, NVCV_GPU, 1), "Error allocating NvCVImage");
            CHECK_MAXINE(NvCVImage_Alloc(&effect.gpu_buf_dst_[1], width_, height_, NVCV_A, NVCV_U8, NVCV_PLANAR, NVCV_GPU, 1), "Error allocating NvCVImage");

            set_bg_color(comp_mode_);

            for (int s = 0; s < max_num_streams_; s++) {
                NvVFX_StateObjectHandle state;
                CHECK_MAXINE(NvVFX_AllocateState(effect.handle_, &state), "Error allocating state");
                state_array_.push_back(state);
            }

            unsigned int model_batch;
            CHECK_MAXINE(NvVFX_GetU32(effect.handle_, NVVFX_MODEL_BATCH, &model_batch), "Error getting model batch");
            batch_of_states_ = (NvVFX_StateObjectHandle*)malloc(sizeof(NvVFX_StateObjectHandle) * model_batch);
            if (batch_of_states_ == nullptr) {
                BMFLOG_NODE(BMF_ERROR, node_id_) << "Error allocating batch of states.";
                return NVCV_ERR_MEMORY;
            }

            CHECK_MAXINE(NvVFX_SetImage(effect.handle_, NVVFX_INPUT_IMAGE, &effect.gpu_buf_src_[0]), "Error setting src image");
            CHECK_MAXINE(NvVFX_SetImage(effect.handle_, NVVFX_OUTPUT_IMAGE, &effect.gpu_buf_dst_[0]), "Error setting dst image");
            CHECK_MAXINE(NvVFX_SetCudaStream(effect.handle_, NVVFX_CUDA_STREAM, stream_), "Error setting cuda stream");
                CHECK_MAXINE(NvVFX_Load(effect.handle_), "Error Loading");
        }
        else if (!strcmp(effect_name.c_str(), NVVFX_FX_BGBLUR)) {
            auto &green_screen = vfx_effects_[i - 1];
            CHECK_MAXINE(NvCVImage_Alloc(&effect.gpu_buf_dst_[0], width_, height_, NVCV_BGR, NVCV_U8, NVCV_CHUNKY, NVCV_GPU, 1), "Error allocating NvCVImage");

            CHECK_MAXINE(NvVFX_SetImage(effect.handle_, NVVFX_INPUT_IMAGE_0, &green_screen.gpu_buf_src_[0]), "Error setting blur input image 0.");
            CHECK_MAXINE(NvVFX_SetImage(effect.handle_, NVVFX_INPUT_IMAGE_1, &green_screen.gpu_buf_dst_[0]), "Error setting blur input image 1.");
            CHECK_MAXINE(NvVFX_SetImage(effect.handle_, NVVFX_OUTPUT_IMAGE, &effect.gpu_buf_dst_[0]), "Error setting blur output image.");
            CHECK_MAXINE(NvVFX_SetCudaStream(effect.handle_, NVVFX_CUDA_STREAM, stream_), "Error setting cuda stream");
            CHECK_MAXINE(NvVFX_Load(effect.handle_), "Error loading blur effect.");
        }
        else if (!strcmp(effect_name.c_str(), NVVFX_FX_DENOISING)) {
            CHECK_MAXINE(NvCVImage_Alloc(&effect.gpu_buf_src_[0], width_, height_, NVCV_BGR, NVCV_F32, NVCV_PLANAR, NVCV_GPU, 1), "Error allocating NvCVImage"); 
            CHECK_MAXINE(NvCVImage_Alloc(&effect.gpu_buf_dst_[0], width_, height_, NVCV_BGR, NVCV_F32, NVCV_PLANAR, NVCV_GPU, 1), "Error allocating NvCVImage");
            CHECK_MAXINE(NvVFX_SetImage(effect.handle_, NVVFX_INPUT_IMAGE, &effect.gpu_buf_src_[0]), "Error setting src image");
            CHECK_MAXINE(NvVFX_SetImage(effect.handle_, NVVFX_OUTPUT_IMAGE, &effect.gpu_buf_dst_[0]), "Error setting dst image");
            CHECK_MAXINE(NvVFX_SetCudaStream(effect.handle_, NVVFX_CUDA_STREAM, stream_), "Error setting cuda stream");
            CHECK_MAXINE(NvVFX_Load(effect.handle_), "Error Loading");
        }
    }

    // wrapper output frame
    auto rgb_info = hmp::PixelInfo(hmp::PF_RGB24, hmp::CS_BT709, hmp::CR_MPEG);
    frm = hmp::Frame(dst_width, dst_height, rgb_info, kCUDA);
    vframe_rgb24_out_ = VideoFrame(frm);
    CHECK_MAXINE(NvCVImage_Init(&vfx_dst_, dst_width, dst_height, dst_width * 3, frm.plane_data(0), NVCV_RGB, NVCV_U8, NVCV_INTERLEAVED, NVCV_GPU), "Error initing NvCVImage");

    has_allocated_ = true;

    return vfx_status;
}

NvCV_Status MaxineModule::composition(NvCVImage &src, NvCVImage &dst, CompMode mode)
{
    NvCV_Status status = NVCV_SUCCESS;
    switch (mode) {
        case compNone:
            NvCVImage_Transfer(&vfx_dst_, &dst, 1.f, stream_, &vfx_tmp_);
            break;
        case compGreen:
            NvCVImage_CompositeOverConstant(&src, &dst, bg_color_, &vfx_dst_, stream_); 
            break;
        case compWhite:
            NvCVImage_CompositeOverConstant(&src, &dst, bg_color_, &vfx_dst_, stream_); 
            break;
        case compBG: {
            NvCVImage_Composite(&src, &vfx_bg_, &dst, &vfx_dst_, stream_);
        } break;
    }
    return status;
}

int MaxineModule::process(Task &task) {
    PacketQueueMap &input_queue_map = task.get_inputs();
    PacketQueueMap::iterator it;

    // process all input queues
    for (it = input_queue_map.begin(); it != input_queue_map.end(); it++) { // input stream label int label = it->first; // input packet queue Packet pkt; // process all packets in one input queue
        // input stream label
        int label = it->first;

        // input packet queue
        Packet pkt;
        while (task.pop_packet_from_input_queue(label, pkt)) {
            // Get a input packet

            // if packet is eof, set module done
            if (pkt.timestamp() == BMF_EOF) {
                task.set_timestamp(DONE);
                task.fill_output_packet(label, Packet::generate_eof_packet());
                return 0;
            }

            // Get packet data
            // Here we should know the data type in packet
            auto vframe = pkt.get<VideoFrame>();

            // YUV420 -> RGB24
            hmp::PixelInfo rgb_info{hmp::PF_RGB24, hmp::CS_BT709, hmp::CR_MPEG};
            VideoFrame vframe_rgb24_in{vframe.width(), vframe.height(), rgb_info, kCUDA};
            
            hmp::Tensor rgb_tensor = vframe_rgb24_in.frame().plane(0);
            hmp::img::yuv_to_rgb(rgb_tensor, vframe.frame().data(),
                                 vframe.frame().pix_info(), hmp::kNHWC);
            
            CHECK_MAXINE(vfx_setup(vframe_rgb24_in), "Error setup");

            // RGB24 -> Formats needed by specific effect
            for (int i = 0; i < num_effects_; i++) {
                if (i == 0) {
                    if (vfx_effects_[i].input_in_rgb_range_) {
                        CHECK_MAXINE(NvCVImage_Transfer(&vfx_src_, &vfx_effects_[i].gpu_buf_src_[0], 1.f, stream_, &vfx_tmp_), "Error transferring image");
                    }
                    else {
                        CHECK_MAXINE(NvCVImage_Transfer(&vfx_src_, &vfx_effects_[i].gpu_buf_src_[0], 1.f / 255.f, stream_, &vfx_tmp_), "Error transferring image");
                    }
                }
                else {
                    float scale = 1.f;
                    if (vfx_effects_[i - 1].input_in_rgb_range_ == vfx_effects_[i].input_in_rgb_range_) {
                        scale = 1.f;
                    }
                    else if (vfx_effects_[i - 1].input_in_rgb_range_ && !vfx_effects_[i].input_in_rgb_range_) {
                        scale = 1.f / 255.f;
                    }
                    else {
                        scale = 255.f;
                    }
                    if (strcmp(vfx_effects_[i].name_.c_str(), NVVFX_FX_BGBLUR)) {
                        CHECK_MAXINE(NvCVImage_Transfer(&vfx_effects_[i - 1].gpu_buf_dst_[0], &vfx_effects_[i].gpu_buf_src_[0], scale, stream_, &vfx_tmp_), "Error transferring image");
                    }
                }

                if (!strcmp(vfx_effects_[i].name_.c_str(), NVVFX_FX_GREEN_SCREEN)) {
                    // TODO What's the relation between batch_of_states and state_array
                    batch_of_states_[0] = state_array_[0];
                    CHECK_MAXINE(NvVFX_SetStateObjectHandleArray(vfx_effects_[i].handle_, NVVFX_STATE, batch_of_states_), "Error seeting state object");
                }

                CHECK_MAXINE(NvVFX_Run(vfx_effects_[i].handle_, 0), "Error running");
            }

            // Effect output formats -> RGB24
            if (vfx_effects_[num_effects_ - 1].input_in_rgb_range_) {
                if (!strcmp(vfx_effects_[num_effects_ - 1].name_.c_str(), NVVFX_FX_GREEN_SCREEN)) {
                    CHECK_MAXINE(NvCVImage_Transfer(&vfx_effects_[num_effects_ - 1].gpu_buf_dst_[0], &vfx_effects_[num_effects_ - 1].gpu_buf_dst_[1], 1.f, stream_, &vfx_tmp_), "Error transferring image");
                    CHECK_MAXINE(composition(vfx_src_, vfx_effects_[num_effects_ - 1].gpu_buf_dst_[1], comp_mode_), "Error in composition");
                }
                else {
                    CHECK_MAXINE(NvCVImage_Transfer(&vfx_effects_[num_effects_ - 1].gpu_buf_dst_[0], &vfx_dst_, 1.f, stream_, &vfx_tmp_), "Error transferring image");
                }
            }
            else {
                CHECK_MAXINE(NvCVImage_Transfer(&vfx_effects_[num_effects_ - 1].gpu_buf_dst_[0], &vfx_dst_, 255.f, stream_, &vfx_tmp_), "Error transferring image");
            }

            // RGB24 -> YUV420P
            hmp::PixelInfo nv12_info{hmp::PF_NV12, hmp::CS_BT470BG, hmp::CR_MPEG};
            VideoFrame vframe_nv12 = vframe_rgb24_out_.reformat(nv12_info);

            vframe_nv12.copy_props(vframe);
            auto output_pkt = Packet(vframe_nv12);
            task.fill_output_packet(label, output_pkt);
        }
    }
    return 0;
}
REGISTER_MODULE_CLASS(MaxineModule)