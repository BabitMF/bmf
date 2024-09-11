#if (defined(__ANDROID__) || defined(__OHOS__)) &&                             \
    defined(BMF_LITE_ENABLE_DENOISE)
#include "denoise_algorithm.h"
#include "algorithm/bmf_video_frame.h"
#include "common/error_code.h"
#include "opengl/denoise/denoise.h"
#include <iostream>
namespace bmf_lite {
struct DenoiseInitParam {
    int algorithm_type = -1; // algorithm_type
    // int scale_mode = -1;     // process 1.5 or 2
    int backend = -1;
    int process_mode = -1;           // process image or video
    int max_width = -1;              // max input width
    int max_height = -1;             // max input height
    std::string license_module_name; // tob license name
    std::string program_cache_dir;   // save program cache dir
    std::string library_path;
    // int sharp_levels = -1;   // mutlti sharp level
    // std::string weight_path; // multi sharp super resolution weight
    bool operator==(const DenoiseInitParam &param) {
        if (algorithm_type != param.algorithm_type)
            return false;
        if (backend != param.backend)
            return false;
        if (process_mode != param.process_mode)
            return false;
        return true;
    }
};
class DenoiseProcessParam {
  public:
    float *mvp = NULL;
};
class DenoiseImpl {
  public:
    struct DenoiseInitParam init_param_;
    //        struct DenoiseProcessParam process_param_;
    std::shared_ptr<opengl::Denoise> denoise_;
    std::shared_ptr<bmf_lite::VideoBufferMultiPool> video_pool_;
    VideoFrame out_frame_;

    DenoiseImpl() {};
    ~DenoiseImpl() {};
    int parseInitParam(Param param, DenoiseInitParam &init_param) {
        if (param.getInt("algorithm_type", init_param.algorithm_type) != 0) {
            return BMF_LITE_StsBadArg;
        }
        if (param.getInt("backend", init_param.backend) != 0) {
            return BMF_LITE_StsBadArg;
        }
        if (param.getInt("process_mode", init_param.process_mode) != 0) {
            return BMF_LITE_StsBadArg;
        }
        if (param.getInt("max_width", init_param.max_width) != 0) {
            return BMF_LITE_StsBadArg;
        }
        if (param.getInt("max_height", init_param.max_height) != 0) {
            return BMF_LITE_StsBadArg;
        }
        if (param.getString("license_module_name",
                            init_param.license_module_name) != 0) {
            return BMF_LITE_StsBadArg;
        }
        if (param.getString("program_cache_dir",
                            init_param.program_cache_dir) != 0) {
            return BMF_LITE_StsBadArg;
        }
        return 0;
    }
    int parseProcessParam(Param param, DenoiseProcessParam &process_param) {
        return 0;
    }
    int setParam(Param param) {
        struct DenoiseInitParam init_param;
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
    int setParam(DenoiseInitParam param) {
        if (init_param_ == param) {
            // do nothing
            // std::cout<<"setParam"<<" set do nothing"<<std::endl;
            return 0;
        } else {
            //                LOGI("DenoiseImpl setParam");
            int ops_result = -1;
            denoise_.reset();
            denoise_ = std::make_shared<opengl::Denoise>();
            ops_result = denoise_->init(init_param_.program_cache_dir);
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
            init_param_ = param;
            return 0;
        }
    }
    int preProcess(VideoFrame frame, DenoiseProcessParam process_param) {

        return 0;
    }
    int createProcessOutVideoFrame(VideoFrame &frame,
                                   DenoiseProcessParam process_param) {
        return 0;
    }
    int processVideoFrame(VideoFrame frame, Param param) {
        //            LOGI("DenoiseImpl processVideoFrame");
        DenoiseProcessParam process_param;
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
                          DenoiseProcessParam process_param) {
        int width = in_frame.buffer()->width();
        int height = in_frame.buffer()->height();
        int in_tex = (long)(in_frame.buffer()->data());
        int out_tex = (long)(out_frame.buffer()->data());
        //            float *mvp = process_param.mvp;
        //            float cm_[9] = {1, -0.00093, 1.401687, 1, -0.3437,
        //            -0.71417, 1, 1.77216, 0.00099}; float co_[3] = {0, -0.5,
        //            -0.5};
        int ops_res = denoise_->run(in_tex, out_tex, width, height);
        return ops_res;
    }
    int getVideoFrameOutput(VideoFrame &frame, Param &param) {
        frame = out_frame_;
        out_frame_ = VideoFrame();
        return 0;
    }
    int unInit() { return 0; };
};

DenoiseAlgorithm::DenoiseAlgorithm() {};
DenoiseAlgorithm::~DenoiseAlgorithm() {};
int DenoiseAlgorithm::setParam(Param param) {
    if (impl_ == NULL) {
        impl_ = std::make_shared<DenoiseImpl>();
    }
    return impl_->setParam(param);
}
int DenoiseAlgorithm::processVideoFrame(VideoFrame frame, Param param) {
    return impl_->processVideoFrame(frame, param);
}
int DenoiseAlgorithm::getVideoFrameOutput(VideoFrame &frame, Param &param) {
    return impl_->getVideoFrameOutput(frame, param);
}
int DenoiseAlgorithm::unInit() { return impl_->unInit(); };
int DenoiseAlgorithm::processMultiVideoFrame(
    std::vector<VideoFrame> videoframes, Param param) {
    return BMF_LITE_StsFuncNotImpl;
}
int DenoiseAlgorithm::getMultiVideoFrameOutput(
    std::vector<VideoFrame> &videoframes, Param &param) {
    return BMF_LITE_StsFuncNotImpl;
}
int DenoiseAlgorithm::getProcessProperty(Param &param) {
    return BMF_LITE_StsFuncNotImpl;
}

int DenoiseAlgorithm::setInputProperty(Param attr) {
    return BMF_LITE_StsFuncNotImpl;
}

int DenoiseAlgorithm::getOutputProperty(Param &attr) {
    return BMF_LITE_StsFuncNotImpl;
}
} // namespace bmf_lite
#endif