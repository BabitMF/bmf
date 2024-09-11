#if (defined(__ANDROID__) || defined(__OHOS__)) &&                             \
    defined(BMF_LITE_ENABLE_SUPER_RESOLUTION)
#include "super_resolution_algorithm.h"
#include "algorithm/bmf_video_frame.h"
#include "common/error_code.h"
#include "opengl/sr/sr.h"
#include <iostream>
namespace bmf_lite {
struct SuperResolutionInitParam {
    int algorithm_type = -1;    // algorithm_type
    int algorithm_version = -1; // algorithm_version
    int scale_mode = -1;        // process 1.5 or 2
    int backend = -1;
    int process_mode = -1;           // process image or video
    int max_width = -1;              // max input width
    int max_height = -1;             // max input height
    std::string license_module_name; // tob license name
    std::string program_cache_dir;   // save program cache dir
    std::string library_path;
    int sharp_levels = -1;   // mutlti sharp level
    std::string weight_path; // multi sharp super resolution weight
    bool operator==(const SuperResolutionInitParam &param) {
        if (algorithm_type == param.algorithm_type &&
            algorithm_version == param.algorithm_version &&
            scale_mode == param.scale_mode)
            return true;
        return false;
    }
};
class SuperResolutionProcessParam {
  public:
    int scale_mode = 0;
    int sharp_level = 0;
    float *mvp = NULL;
};
class SuperResolutionImpl {
  public:
    struct SuperResolutionInitParam init_param_;
    struct SuperResolutionProcessParam process_param_;

    std::shared_ptr<opengl::Sr> sr_hp_;

    std::shared_ptr<bmf_lite::VideoBufferMultiPool> video_pool_;
    VideoFrame out_frame_;

    SuperResolutionImpl() {};
    ~SuperResolutionImpl() {};
    int parseInitParam(Param param, SuperResolutionInitParam &init_param) {
        if (param.getInt("algorithm_type", init_param.algorithm_type) != 0) {
            return BMF_LITE_StsBadArg;
        }
        if (param.getInt("scale_mode", init_param.scale_mode) != 0) {
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
        if (param.getInt("sharp_levels", init_param.sharp_levels) != 0) {
            return BMF_LITE_StsBadArg;
        }
        if (param.getString("weight_path", init_param.weight_path) != 0) {
            return BMF_LITE_StsBadArg;
        }
        return 0;
    }
    int parseProcessParam(Param param,
                          SuperResolutionProcessParam &process_param) {
        if (param.getInt("sharp_level", process_param.sharp_level) != 0) {
            return BMF_LITE_StsBadArg;
        }
        if (param.getInt("scale_mode", process_param.scale_mode) != 0) {
            return BMF_LITE_StsBadArg;
        }
        return 0;
    }
    int setParam(Param param) {
        struct SuperResolutionInitParam init_param;
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
    int setParam(SuperResolutionInitParam param) {
        if (init_param_ == param) {
            // do nothing
            // std::cout<<"setParam"<<" set do nothing"<<std::endl;
            return 0;
        } else {
            int hydra_result = -1;
            sr_hp_.reset();
            sr_hp_ = std::make_shared<opengl::Sr>();
            hydra_result = sr_hp_->init(init_param_.program_cache_dir);
            if (0 != hydra_result) {
                return BMF_LITE_OpsError;
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
    int preProcess(VideoFrame frame,
                   SuperResolutionProcessParam process_param) {
        return 0;
    }
    int createProcessOutVideoFrame(VideoFrame &frame,
                                   SuperResolutionProcessParam process_param) {
        return 0;
    }
    int processVideoFrame(VideoFrame frame, Param param) {
        SuperResolutionProcessParam process_param;
        int res = parseProcessParam(param, process_param);
        if (res < 0) {
            return res;
        }
        std::shared_ptr<bmf_lite::VideoBuffer> video_buffer = frame.buffer();
        bmf_lite::HardwareDataInfo hardware_data_info =
            video_buffer->hardwareDataInfo();
        hardware_data_info.mutable_flag = 1;
        std::shared_ptr<bmf_lite::VideoBuffer> output_video_buffer =
            video_pool_->acquireObject(video_buffer->width() * 2,
                                       video_buffer->height() * 2,
                                       hardware_data_info);
        if (output_video_buffer == NULL) {
            return BMF_LITE_StsNoMem;
        }
        out_frame_ = VideoFrame(output_video_buffer);
        // std::cout<<"SuperResolutionAlgorithm::processVideoFrame
        // "<<"processVideoFrame"<<std::endl;
        res = processVideoFrame(frame, out_frame_, process_param);
        if (res < 0) {
            return res;
        }
        // std::cout<<"SuperResolutionAlgorithm::processVideoFrame
        // "<<"processVideoFrame end"<<std::endl;
        return 0;
    }
    int processVideoFrame(VideoFrame in_frame, VideoFrame out_frame,
                          SuperResolutionProcessParam process_param) {
        int width = in_frame.buffer()->width();
        int height = in_frame.buffer()->height();
        int in_tex = (long)(in_frame.buffer()->data());
        int out_tex = (long)(out_frame.buffer()->data());
        int sharp_level = process_param.sharp_level;
        float *mvp = process_param.mvp;
        float cm_[9] = {1,        -0.00093, 1.401687, 1,      -0.3437,
                        -0.71417, 1,        1.77216,  0.00099};
        float co_[3] = {0, -0.5, -0.5};
        int ops_res = sr_hp_->run(in_tex, out_tex, width, height);
        if (0 != ops_res) {
            return BMF_LITE_OpsError;
        }
        return 0;
    }
    int getVideoFrameOutput(VideoFrame &frame, Param &param) {
        frame = out_frame_;
        out_frame_ = VideoFrame();
        return 0;
    }
    int unInit() { return 0; };
};

SuperResolutionAlgorithm::SuperResolutionAlgorithm() {};
SuperResolutionAlgorithm::~SuperResolutionAlgorithm() {};
int SuperResolutionAlgorithm::setParam(Param param) {
    if (impl_ == NULL) {
        impl_ = std::make_shared<SuperResolutionImpl>();
    }
    return impl_->setParam(param);
}
int SuperResolutionAlgorithm::processVideoFrame(VideoFrame frame, Param param) {
    return impl_->processVideoFrame(frame, param);
}
int SuperResolutionAlgorithm::getVideoFrameOutput(VideoFrame &frame,
                                                  Param &param) {
    return impl_->getVideoFrameOutput(frame, param);
}
int SuperResolutionAlgorithm::unInit() { return impl_->unInit(); };
int SuperResolutionAlgorithm::processMultiVideoFrame(
    std::vector<VideoFrame> videoframes, Param param) {
    return BMF_LITE_StsFuncNotImpl;
}
int SuperResolutionAlgorithm::getMultiVideoFrameOutput(
    std::vector<VideoFrame> &videoframes, Param &param) {
    return BMF_LITE_StsFuncNotImpl;
}
int SuperResolutionAlgorithm::getProcessProperty(Param &param) {
    return BMF_LITE_StsFuncNotImpl;
}

int SuperResolutionAlgorithm::setInputProperty(Param attr) {
    return BMF_LITE_StsFuncNotImpl;
}

int SuperResolutionAlgorithm::getOutputProperty(Param &attr) {
    return BMF_LITE_StsFuncNotImpl;
}
} // namespace bmf_lite
#endif