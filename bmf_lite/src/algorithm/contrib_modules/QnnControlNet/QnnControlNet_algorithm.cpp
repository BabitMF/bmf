#if (defined(__ANDROID__) || defined(__OHOS__)) &&                             \
    defined(BMF_LITE_ENABLE_TEX_GEN_PIC)
#include "QnnControlNet_algorithm.h"
#include "QnnControlNetPipeLine.h"
#include "algorithm/bmf_video_frame.h"
#include "common/error_code.h"
#include "gl_texture_transformer.h"
#include <iostream>
namespace bmf_lite {
struct ControlNetInitParam {
    int algorithm_type = -1;         // algorithm_type
    int process_mode = -1;           // process image or video
    std::string license_module_name; // tob license name
    std::string qnn_htp_library_path;
    std::string qnn_system_library_path;
    std::string ld_library_path;
    std::string adsp_system_library_path;

    std::string tokenizer_path;
    std::string unet_path;
    std::string text_encoder_path;
    std::string vae_path;
    std::string control_net_path;

    bool operator==(const ControlNetInitParam &param) {
        if (algorithm_type != param.algorithm_type)
            return false;
        if (process_mode != param.process_mode)
            return false;
        return true;
    }
};
class ControlNetProcessParam {
  public:
    float *mvp = NULL;
    std::string positive_prompt_en = "";
    std::string negative_prompt_en = "";
    int seed = 0;
    int step = 1;
    int new_prompt = 1;
};
class QNNControlNetImpl {
  public:
    struct ControlNetInitParam init_param_;
    struct ControlNetProcessParam process_param_;
    std::shared_ptr<QnnControlNetPipeline> control_net_;
    std::shared_ptr<bmf_lite::VideoBufferMultiPool> video_pool_;
    VideoFrame out_frame_;
    std::shared_ptr<bmf_lite::GLTextureTransformer> trans_ = nullptr;
    float *out_data_ptr_ = nullptr;
    float *in_data_ptr_ = nullptr;
    std::shared_ptr<VideoBuffer> trans_in_video_buffer_ = nullptr;
    std::shared_ptr<VideoBuffer> trans_out_video_buffer_ = nullptr;
    const int MODEL_WIDTH = 512;
    const int MODEL_HEIGHT = 512;
    const char *LD_LIBRARY_PATH =
        "/data/user/0/com.bmf.lite.app/files:/vendor/dsp/cdsp:/vendor/lib64:/"
        "vendor/dsp/dsp:/vendor/dsp/images";
    const char *ADSP_LIBRARY_PATH =
        "/data/user/0/com.bmf.lite.app/files;/vendor/dsp/cdsp;/vendor/lib/rfsa/"
        "adsp;/system/lib/rfsa/adsp;/vendor/dsp/dsp;/vendor/dsp/images;/dsp";
    const char *QNN_HTP_DEFAULT_PATH =
        "/data/user/0/com.bmf.lite.app/files/libQnnHtp.so";
    const char *QNN_SYSTEM_DEFAULT_PATH =
        "/data/user/0/com.bmf.lite.app/files/libQnnSystem.so";
    QNNControlNetImpl() {};
    ~QNNControlNetImpl() {};
    int parseInitParam(Param param, ControlNetInitParam &init_param) {
        if (param.getInt("algorithm_type", init_param.algorithm_type) != 0) {
            return BMF_LITE_StsBadArg;
        }

        if (param.getInt("process_mode", init_param.process_mode) != 0) {
            return BMF_LITE_StsBadArg;
        }
        if (param.getString("qnn_htp_library_path",
                            init_param.qnn_htp_library_path) != 0) {
            return BMF_LITE_StsBadArg;
        }
        if (param.getString("qnn_system_library_path",
                            init_param.qnn_system_library_path) != 0) {
            return BMF_LITE_StsBadArg;
        }
        if (param.getString("ld_library_path", init_param.ld_library_path) !=
            0) {
            return BMF_LITE_StsBadArg;
        }
        if (param.getString("adsp_system_library_path",
                            init_param.adsp_system_library_path) != 0) {
            return BMF_LITE_StsBadArg;
        }

        if (param.getString("tokenizer_path", init_param.tokenizer_path) != 0) {
            return BMF_LITE_StsBadArg;
        }
        if (param.getString("unet_path", init_param.unet_path) != 0) {
            return BMF_LITE_StsBadArg;
        }
        if (param.getString("text_encoder_path",
                            init_param.text_encoder_path) != 0) {
            return BMF_LITE_StsBadArg;
        }
        if (param.getString("vae_path", init_param.vae_path) != 0) {
            return BMF_LITE_StsBadArg;
        }
        if (param.getString("control_net_path", init_param.control_net_path) !=
            0) {
            return BMF_LITE_StsBadArg;
        }

        return 0;
    }
    int parseProcessParam(Param param, ControlNetProcessParam &process_param) {
        if (param.getString("positive_prompt_en",
                            process_param.positive_prompt_en) != 0) {
            return BMF_LITE_StsBadArg;
        }
        if (param.getString("negative_prompt_en",
                            process_param.negative_prompt_en) != 0) {
            return BMF_LITE_StsBadArg;
        }
        if (param.getInt("seed", process_param.seed) != 0) {
            return BMF_LITE_StsBadArg;
        }
        if (param.getInt("step", process_param.step) != 0) {
            return BMF_LITE_StsBadArg;
        }
        if (param.getInt("new_prompt", process_param.new_prompt) != 0) {
            return BMF_LITE_StsBadArg;
        }
        return 0;
    }
    int setParam(Param param) {
        BMFLITE_LOGI("controlnet", "QNNControlNetImpl setParam  0");
        struct ControlNetInitParam init_param;
        int res = parseInitParam(param, init_param);
        if (res < 0) {
            return res;
        }
        res = setParam(init_param);
        if (res < 0) {
            return res;
        }
        return 0;
    }
    int setParam(ControlNetInitParam param) {
        if (init_param_ == param) {
            // do nothing
            // std::cout<<"setParam"<<" set do nothing"<<std::endl;
            return 0;
        } else {
            // std::cout<<"QNNControlNetImpl setParam "<<std::endl;
            int ops_result = -1;
            control_net_ = std::make_shared<QnnControlNetPipeline>();

            auto rest =
                setenv("LD_LIBRARY_PATH", param.ld_library_path.c_str(), true);
            rest = setenv("ADSP_LIBRARY_PATH",
                          param.adsp_system_library_path.c_str(), true);
            ops_result = control_net_->init(
                param.qnn_htp_library_path.c_str(),
                param.qnn_system_library_path.c_str(),
                param.tokenizer_path.c_str(), param.unet_path.c_str(),
                param.text_encoder_path.c_str(), param.vae_path.c_str(),
                param.control_net_path.c_str());
            if (0 != ops_result) {
                return ops_result;
            }
            bmf_lite::MemoryType memory_type =
                bmf_lite::MemoryType::kOpenGLTexture2d;
            std::shared_ptr<bmf_lite::HWDeviceContext> device_context;
            bmf_lite::HWDeviceType device_type = bmf_lite::kHWDeviceTypeEGLCtx;
            bmf_lite::HardwareDeviceCreateInfo create_info;
            create_info.device_type = device_type;
            bmf_lite::EGLContextInfo context_info;
            context_info.egl_context = NULL;
            context_info.egl_display = NULL;
            context_info.egl_draw_surface = NULL;
            context_info.egl_read_surface = NULL;
            create_info.context_info = &context_info;
            int res = bmf_lite::HWDeviceContextManager::createHwDeviceContext(
                &create_info, device_context);
            if (res < 0) {
                return res;
            }
            std::shared_ptr<bmf_lite::VideoBufferAllocator> allocator =
                bmf_lite::AllocatorManager::getAllocator(memory_type,
                                                         device_context);
            int max_size = 2;
            video_pool_ = std::make_shared<bmf_lite::VideoBufferMultiPool>(
                allocator, device_context, max_size);
            if (video_pool_ == NULL) {
                return BMF_LITE_StsNoMem;
            }

            bmf_lite::HardwareDataInfo data_info = {
                bmf_lite::MemoryType::kOpenGLTexture2d,
                bmf_lite::GLES_TEXTURE_RGBA, 0};

            trans_ = std::make_shared<GLTextureTransformer>();
            res = trans_->init(data_info, device_context);
            if (res != BMF_LITE_StsOk) {
                return res;
            }
            if (in_data_ptr_ == nullptr) {
                in_data_ptr_ = new float[MODEL_WIDTH * MODEL_HEIGHT * 3];
            }
            if (out_data_ptr_ == nullptr) {
                out_data_ptr_ = new float[MODEL_WIDTH * MODEL_HEIGHT * 3];
            }

            std::shared_ptr<bmf_lite::HWDeviceContext> device_context_cpu;
            bmf_lite::HWDeviceContextManager::getCurrentHwDeviceContext(
                bmf_lite::kHWDeviceTypeNone, device_context_cpu);
            bmf_lite::HardwareDataInfo data_info_cpu = {
                bmf_lite::MemoryType::kByteMemory, bmf_lite::CPU_RGBFLOAT, 0};
            if (trans_in_video_buffer_ == nullptr) {
                bmf_lite::VideoBufferManager::
                    createTextureVideoBufferFromExistingData(
                        (void *)in_data_ptr_, MODEL_WIDTH, MODEL_HEIGHT,
                        &data_info_cpu, device_context_cpu, NULL,
                        trans_in_video_buffer_);
            }
            if (trans_out_video_buffer_ == nullptr) {
                bmf_lite::VideoBufferManager::
                    createTextureVideoBufferFromExistingData(
                        (void *)in_data_ptr_, MODEL_WIDTH, MODEL_HEIGHT,
                        &data_info_cpu, device_context_cpu, NULL,
                        trans_out_video_buffer_);
            }
            init_param_ = param;
            return 0;
        }
    }
    int preProcess(VideoFrame frame, ControlNetProcessParam process_param) {
        return 0;
    }
    int createProcessOutVideoFrame(VideoFrame &frame,
                                   ControlNetProcessParam process_param) {
        return 0;
    }
    int processVideoFrame(VideoFrame frame, Param param) {
        ControlNetProcessParam process_param;
        int res = parseProcessParam(param, process_param);
        if (res < 0) {
            return res;
        }
        std::shared_ptr<bmf_lite::VideoBuffer> video_buffer = frame.buffer();
        bmf_lite::HardwareDataInfo hardware_data_info =
            video_buffer->hardwareDataInfo();
        hardware_data_info.mutable_flag = 1;
        std::shared_ptr<bmf_lite::VideoBuffer> output_video_buffer =
            video_pool_->acquireObject(video_buffer->width(),
                                       video_buffer->height(),
                                       hardware_data_info);
        if (output_video_buffer == NULL) {
            return BMF_LITE_StsNoMem;
        }

        out_frame_ = VideoFrame(output_video_buffer);
        res = processVideoFrame(frame, out_frame_, process_param);
        if (res < 0) {
            return res;
        }
        return 0;
    }
    int processVideoFrame(VideoFrame in_frame, VideoFrame out_frame,
                          ControlNetProcessParam process_param) {
        int width = in_frame.buffer()->width();
        int height = in_frame.buffer()->height();
        int in_tex = (long)(in_frame.buffer()->data());
        int out_tex = (long)(out_frame.buffer()->data());
        int ret = trans_->transTexture2Memory(in_frame.buffer(),
                                              trans_in_video_buffer_);
        if (ret != BMF_LITE_StsOk) {
            return ret;
        }
        if (process_param.new_prompt == 1) {
            int ops_res = control_net_->tokenize(
                process_param.positive_prompt_en,
                process_param.negative_prompt_en,
                (float *)(trans_in_video_buffer_->data()), 20,
                process_param.seed);
            if (ops_res != BMF_LITE_StsOk) {
                return ops_res;
            }
        }
        int ops_res = control_net_->step(
            (float *)(trans_out_video_buffer_->data()), process_param.step);
        if (ops_res != BMF_LITE_StsOk) {
            return ops_res;
        }
        ret = trans_->transMemory2Texture(trans_out_video_buffer_,
                                          out_frame.buffer());
        return ret;
    }
    int getVideoFrameOutput(VideoFrame &frame, Param &param) {
        frame = out_frame_;
        out_frame_ = VideoFrame();
        return 0;
    }
    int unInit() {
        if (out_data_ptr_ != nullptr) {
            delete[] out_data_ptr_;
            out_data_ptr_ = nullptr;
        }
        if (in_data_ptr_ != nullptr) {
            delete[] in_data_ptr_;
            in_data_ptr_ = nullptr;
        }

        return 0;
    };

  private:
    std::shared_ptr<QnnHTPRuntime> htp_runtime_;
    std::shared_ptr<QnnModel> unet, text_encoder, vae, control_net;
    std::shared_ptr<QnnTensorData> tokens, text_latents, bad_text_latents,
        noise_latent, time_embeddings, canny_image, noise_latent_out, vae_in,
        out_image;
    std::vector<std::shared_ptr<QnnTensorData>> controlnet_blks;
};

QNNControlNetAlgorithm::QNNControlNetAlgorithm() {};
QNNControlNetAlgorithm::~QNNControlNetAlgorithm() {};
int QNNControlNetAlgorithm::setParam(Param param) {
    if (impl_ == NULL) {
        impl_ = std::make_shared<QNNControlNetImpl>();
    }
    return impl_->setParam(param);
}
int QNNControlNetAlgorithm::processVideoFrame(VideoFrame frame, Param param) {
    return impl_->processVideoFrame(frame, param);
}
int QNNControlNetAlgorithm::getVideoFrameOutput(VideoFrame &frame,
                                                Param &param) {
    return impl_->getVideoFrameOutput(frame, param);
}
int QNNControlNetAlgorithm::unInit() { return impl_->unInit(); };
int QNNControlNetAlgorithm::processMultiVideoFrame(
    std::vector<VideoFrame> videoframes, Param param) {
    return 0;
};

int QNNControlNetAlgorithm::getMultiVideoFrameOutput(
    std::vector<VideoFrame> &videoframes, Param &param) {
    return 0;
};

int QNNControlNetAlgorithm::getProcessProperty(Param &param) { return 0; };

int QNNControlNetAlgorithm::setInputProperty(Param attr) { return 0; };

int QNNControlNetAlgorithm::getOutputProperty(Param &attr) { return 0; };
} // namespace bmf_lite
#endif