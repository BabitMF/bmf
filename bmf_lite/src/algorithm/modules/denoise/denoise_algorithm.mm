#include "denoise_algorithm.h"
#include "algorithm/bmf_video_frame.h"
#include "common/error_code.h"
#include "media/video_buffer/metal_texture_video_buffer/mtl_device_context.h"
#include "media/video_buffer/transform/cvpixelbuffer_transformer.h"
#include "metal/denoise/denoise.h"
#include "metal/metal_helper.h"
#import <AVFoundation/AVFoundation.h>

#if defined(__APPLE__) && defined(BMF_LITE_ENABLE_DENOISE)

namespace bmf_lite {

struct DenoiseInitParam {
  int algorithm_type = 0;
  int pool_size = 2;
  std::string library_path;
  bool operator==(const DenoiseInitParam &param) {
    if (algorithm_type == param.algorithm_type) {
      return true;
    }
    return false;
  }
};

struct DenoiseProcessParam {};

class DenoiseImpl {
public:
  int parseInitParam(Param param, DenoiseInitParam &init_param) {
    if (param.getInt("algorithm_type", init_param.algorithm_type) != 0) {
      return BMF_LITE_StsBadArg;
    }

    if (param.getString("library_path", init_param.library_path) != 0) {
      return BMF_LITE_StsBadArg;
    }
    return BMF_LITE_StsOk;
  }

  int parseProcessParam(const Param &param,
                        DenoiseProcessParam &process_param) {
    return BMF_LITE_StsOk;
  }

  int setParam(Param param) {
    struct DenoiseInitParam init_param;
    int ret = parseInitParam(param, init_param);
    if (ret != BMF_LITE_StsOk) {
      return ret;
    }

    if (init_param_ == init_param && denoise_ != nullptr) {
      return BMF_LITE_StsOk;
    }

    denoise_.reset(new (std::nothrow) metal::Denoise());
    ret = denoise_->init();
    if (ret != BMF_LITE_StsOk) {
      return ret;
    }
    command_queue_ =
        [metal::MetalHelper::instance().mtl_device() newCommandQueue];
    MemoryType memory_type = MemoryType::kCVPixelBuffer;
    std::shared_ptr<MtlDeviceContext> mtl_device_context =
        std::make_shared<MtlDeviceContext>();
    mtl_device_context->create_context();

    std::shared_ptr<VideoBufferAllocator> allocator =
        AllocatorManager::getAllocator(memory_type, mtl_device_context);
    pool_ = std::make_shared<VideoBufferMultiPool>(
        allocator, mtl_device_context, init_param.pool_size);

    HardwareDataInfo hardware_data_info_in;
    hardware_data_info_in.mem_type = MemoryType::kCVPixelBuffer;

    HardwareDataInfo hardware_data_info_out;
    hardware_data_info_out.mem_type = MemoryType::kMultiMetalTexture;

    trans_ = std::make_shared<CvPixelBufferTransformer>();
    ret = trans_->init(hardware_data_info_in, mtl_device_context,
                       hardware_data_info_out);
    if (ret != BMF_LITE_StsOk) {
      return ret;
    }
    init_param_ = init_param;

    return BMF_LITE_StsOk;
  }

  int processVideoFrame(VideoFrame *in_frame, std::shared_ptr<bmf_lite::VideoBuffer> obuf,
                        const DenoiseProcessParam &process_param) {
    std::shared_ptr<VideoBuffer> ibuf = in_frame->buffer();
    if (ibuf == nullptr) {
      return BMF_LITE_StsBadArg;
    }

    std::shared_ptr<VideoBuffer> temp_ibuf = nullptr;
    int ret = trans_->trans(ibuf, temp_ibuf);
    if (ret != BMF_LITE_StsOk) {
      return ret;
    }

    VideoFrame temp_iframe(temp_ibuf);

    VideoTextureList *temp_multi_data =
        (VideoTextureList *)(temp_iframe.buffer()->data());
    id<MTLTexture> y_tex =
        (__bridge_transfer id<MTLTexture>)(temp_multi_data->texture_list[0]
                                               ->data());
    id<MTLTexture> uv_tex =
        (__bridge_transfer id<MTLTexture>)(temp_multi_data->texture_list[1]
                                               ->data());

    std::shared_ptr<VideoBuffer> temp_obuf = nullptr;
    ret = trans_->trans(obuf, temp_obuf);
    if (ret != BMF_LITE_StsOk) {
      return ret;
    }

    VideoFrame temp_oframe(temp_obuf);

    VideoTextureList *out_multi_data =
        (VideoTextureList *)(temp_oframe.buffer()->data());
    id<MTLTexture> y_otex =
        (__bridge_transfer id<MTLTexture>)(out_multi_data->texture_list[0]
                                               ->data());
    id<MTLTexture> uv_otex =
        (__bridge_transfer id<MTLTexture>)(out_multi_data->texture_list[1]
                                               ->data());

    if (y_tex == nil || uv_tex == nil || y_otex == nil || uv_otex == nil) {
      return BMF_LITE_StsTexTypeError;
    }

    ret = denoise_->run(y_tex, uv_tex, y_otex, uv_otex, command_queue_);
    if (ret != BMF_LITE_StsOk) {
      return ret;
    }
    return BMF_LITE_StsOk;
  }

  DenoiseImpl() {}

  ~DenoiseImpl() {}

  int processVideoFrame(VideoFrame frame, Param &param) {
    DenoiseProcessParam process_param;
    int ret = parseProcessParam(param, process_param);
    if (ret != BMF_LITE_StsOk) {
      return ret;
    }

    std::shared_ptr<bmf_lite::VideoBuffer> video_buffer = frame.buffer();
    if (pool_ == nullptr) {
      return BMF_LITE_OpsError;
    }

    HardwareDataInfo hardware_data_info;
    hardware_data_info.mem_type = MemoryType::kCVPixelBuffer;
    hardware_data_info.internal_format =
        video_buffer->hardwareDataInfo().internal_format;

    // Todo: if have attachment
    int ow = video_buffer->width();
    int oh = video_buffer->height();
    output_video_buffer_ = nullptr;
    output_video_buffer_ = pool_->acquireObject(ow, oh, hardware_data_info);

    if (output_video_buffer_ == nullptr) {
      return BMF_LITE_StsNoMem;
    }

    return processVideoFrame(&frame, output_video_buffer_, process_param);
  }

  int getVideoFrameOutput(VideoFrame &frame, const Param &param) {
    frame = VideoFrame(output_video_buffer_);
    return 0;
  }

private:
  id<MTLCommandQueue> command_queue_ = nil;
  DenoiseInitParam init_param_;
  std::shared_ptr<metal::Denoise> denoise_ = nullptr;
  std::shared_ptr<bmf_lite::CvPixelBufferTransformer> trans_ = nullptr;
  std::shared_ptr<bmf_lite::VideoBufferMultiPool> pool_ = nullptr;
  std::shared_ptr<bmf_lite::VideoBuffer> output_video_buffer_ = nullptr;
};

int DenoiseAlgorithm::setParam(Param param) {
  if (impl_ == nullptr) {
    impl_.reset(new (std::nothrow) DenoiseImpl());
  }
  return impl_->setParam(param);
}

int DenoiseAlgorithm::processVideoFrame(VideoFrame frame, Param param) {
  return impl_->processVideoFrame(frame, param);
}

int DenoiseAlgorithm::getVideoFrameOutput(VideoFrame &frame, Param &param) {
  return impl_->getVideoFrameOutput(frame, param);
}

int DenoiseAlgorithm::processMultiVideoFrame(
    std::vector<VideoFrame> videoframes, Param param) {
  return -1;
}

int DenoiseAlgorithm::getMultiVideoFrameOutput(
    std::vector<VideoFrame> &videoframes, Param &param) {
  return -1;
}

int DenoiseAlgorithm::unInit() { impl_ = nullptr; }

DenoiseAlgorithm::DenoiseAlgorithm() {}

DenoiseAlgorithm::~DenoiseAlgorithm() {}

int DenoiseAlgorithm::getProcessProperty(Param &param) {
  return BMF_LITE_StsFuncNotImpl;
}

int DenoiseAlgorithm::setInputProperty(Param attr) {
  return BMF_LITE_StsFuncNotImpl;
}

int DenoiseAlgorithm::getOutputProperty(Param &attr) {
  return BMF_LITE_StsFuncNotImpl;
}

}

#endif