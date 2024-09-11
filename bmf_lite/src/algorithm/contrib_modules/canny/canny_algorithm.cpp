#if (defined(__ANDROID__) || defined(__OHOS__)) &&                             \
    defined(BMF_LITE_ENABLE_CANNY)
#include "canny_algorithm.h"
#include "algorithm/bmf_video_frame.h"
#include "common/error_code.h"
// #include "opengl/canny/canny.h" //android was not implemented
#include <iostream>
namespace bmf_lite {
struct CannyInitParam {
    int algorithm_type = -1;         // algorithm type
    int backend = -1;                // backend type
    int process_mode = -1;           // process image or video
    int max_width = -1;              // max input width
    int max_height = -1;             // max input height
    std::string license_module_name; // tob license name
    std::string program_cache_dir;   // save program cache dir
    std::string library_path;        // lib path
    bool operator==(const CannyInitParam &param) {
        if (algorithm_type != param.algorithm_type)
            return false;
        if (backend != param.backend)
            return false;
        if (process_mode != param.process_mode)
            return false;
        if (max_width != param.max_width)
            return false;
        return true;
    }
};
class CannyProcessParam {
  public:
    float *mvp = NULL;
};
class CannyImpl {
  public:
    struct CannyInitParam init_param_;
    // std::shared_ptr<opengl::Canny> canny_; //android  was not implemented
    std::shared_ptr<bmf_lite::VideoBufferMultiPool> video_pool_;
    VideoFrame out_frame_;

    CannyImpl() {};
    ~CannyImpl() {};
    int parseInitParam(Param param, CannyInitParam &init_param) {
        return BMF_LITE_UnSupport;
    }
    int parseProcessParam(Param param, CannyProcessParam &process_param) {
        return BMF_LITE_UnSupport;
    }
    int setParam(Param param) { return BMF_LITE_UnSupport; }
    int setParam(CannyInitParam param) { return BMF_LITE_UnSupport; }
    int preProcess(VideoFrame frame, CannyProcessParam process_param) {
        return BMF_LITE_UnSupport;
    }
    int createProcessOutVideoFrame(VideoFrame &frame,
                                   CannyProcessParam process_param) {
        return BMF_LITE_UnSupport;
    }
    int processVideoFrame(VideoFrame frame, Param param) {
        return BMF_LITE_UnSupport;
    }
    int processVideoFrame(VideoFrame in_frame, VideoFrame out_frame,
                          CannyProcessParam process_param) {
        return BMF_LITE_UnSupport;
    }
    int getVideoFrameOutput(VideoFrame &frame, Param &param) {
        return BMF_LITE_UnSupport;
    }
    int unInit() { return BMF_LITE_UnSupport; };
};

CannyAlgorithm::CannyAlgorithm() {};
CannyAlgorithm::~CannyAlgorithm() {};
int CannyAlgorithm::setParam(Param param) { return BMF_LITE_UnSupport; }
int CannyAlgorithm::processVideoFrame(VideoFrame frame, Param param) {
    return BMF_LITE_UnSupport;
}
int CannyAlgorithm::getVideoFrameOutput(VideoFrame &frame, Param &param) {
    return BMF_LITE_UnSupport;
}
int CannyAlgorithm::unInit() { return BMF_LITE_UnSupport; };

int CannyAlgorithm::processMultiVideoFrame(std::vector<VideoFrame> videoframes,
                                           Param param) {
    return 0;
};

int CannyAlgorithm::getMultiVideoFrameOutput(
    std::vector<VideoFrame> &videoframes, Param &param) {
    return BMF_LITE_UnSupport;
};

int CannyAlgorithm::getProcessProperty(Param &param) {
    return BMF_LITE_UnSupport;
};

int CannyAlgorithm::setInputProperty(Param attr) { return BMF_LITE_UnSupport; };

int CannyAlgorithm::getOutputProperty(Param &attr) {
    return BMF_LITE_UnSupport;
};
} // namespace bmf_lite
#endif