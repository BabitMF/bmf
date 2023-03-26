#include <bmf/sdk/ffmpeg_helper.h>
#include "custom_ffmpeg.h"


#define BMF_PROTECT(...) 	            \
	try{                                \
        __VA_ARGS__                     \
    } catch(const std::exception &e){   \
        bmf_set_last_error(e.what());   \
    }


extern "C"
    BMF_API bmf_VideoFrame bmf_vf_from_avframe(const AVFrame *avf)
{
  BMF_PROTECT(
      auto vf = bmf_sdk::ffmpeg::to_video_frame(avf, true);
      return new bmf_sdk::VideoFrame(vf);
  )
  return nullptr;
}

extern "C"
    BMF_API AVFrame* bmf_vf_to_avframe(const bmf_VideoFrame vf)
{
  BMF_PROTECT(
      return bmf_sdk::ffmpeg::from_video_frame(*vf, false);
  )
  return nullptr;
}


extern "C"
    BMF_API bmf_AudioFrame bmf_af_from_avframe(const AVFrame *aaf)
{
  BMF_PROTECT(
      auto af = bmf_sdk::ffmpeg::to_audio_frame(aaf, true);
      return new bmf_sdk::AudioFrame(af);
  )
  return nullptr;
}

extern "C"
    BMF_API AVFrame* bmf_af_to_avframe(const bmf_AudioFrame af)
{
  BMF_PROTECT(
      return bmf_sdk::ffmpeg::from_audio_frame(*af, false);
  )
  return nullptr;
}
