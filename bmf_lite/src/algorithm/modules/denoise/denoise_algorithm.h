#ifndef _BMF_DENOISE_ALGORITHM_H_
#define _BMF_DENOISE_ALGORITHM_H_
#include "algorithm/bmf_algorithm.h"
#include "algorithm/algorithm_interface.h"
#include "media/video_buffer/video_buffer_pool.h"
namespace bmf_lite {

class DenoiseImpl;
class DenoiseAlgorithm : public IAlgorithmInterface {

  public:
    DenoiseAlgorithm();
    virtual ~DenoiseAlgorithm();

    int setParam(Param param);
    int unInit();

    int processVideoFrame(VideoFrame videoframe, Param param);
    int getVideoFrameOutput(VideoFrame &frame, Param &param);

    int processMultiVideoFrame(std::vector<VideoFrame> videoframes,
                               Param param);
    int getMultiVideoFrameOutput(std::vector<VideoFrame> &videoframes,
                                 Param &param);

    int getProcessProperty(Param &param);
    int setInputProperty(Param attr);
    int getOutputProperty(Param &attr);
    std::shared_ptr<DenoiseImpl> impl_;
};
} // namespace bmf_lite
#endif