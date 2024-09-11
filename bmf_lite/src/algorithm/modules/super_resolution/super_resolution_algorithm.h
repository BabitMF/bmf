#ifndef _BMF_SUPER_RESOLUTION_ALGORITHM_H_
#define _BMF_SUPER_RESOLUTION_ALGORITHM_H_
#include "algorithm/algorithm_interface.h"
#include "algorithm/bmf_algorithm.h"
#include "media/video_buffer/video_buffer_pool.h"

namespace bmf_lite {

class SuperResolutionImpl;
class SuperResolutionAlgorithm : public IAlgorithmInterface {
  public:
    SuperResolutionAlgorithm();
    virtual ~SuperResolutionAlgorithm();

    int setParam(Param param) override;
    int unInit();

    int processVideoFrame(VideoFrame frame, Param param) override;
    int getVideoFrameOutput(VideoFrame &frame, Param &param) override;

    int processMultiVideoFrame(std::vector<VideoFrame> videoframes,
                               Param param) override;
    int getMultiVideoFrameOutput(std::vector<VideoFrame> &videoframes,
                                 Param &param) override;

    int getProcessProperty(Param &param) override;
    int setInputProperty(Param attr) override;
    int getOutputProperty(Param &attr) override;

    std::shared_ptr<SuperResolutionImpl> impl_ = nullptr;
};

} // namespace bmf_lite

#endif