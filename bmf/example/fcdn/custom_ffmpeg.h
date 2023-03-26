#ifndef ENGINE_WRAPPER_CUSTOM_FFMPEG_H
#define ENGINE_WRAPPER_CUSTOM_FFMPEG_H

#include <bmf/sdk/bmf_capi.h>

#ifdef __cplusplus
extern "C" {
#endif
#include <libavformat/avformat.h>

struct AVFrame;

BMF_API bmf_VideoFrame bmf_vf_from_avframe(const AVFrame *avf);

BMF_API AVFrame* bmf_vf_to_avframe(const bmf_VideoFrame vf);

BMF_API bmf_AudioFrame bmf_af_from_avframe(const AVFrame *aaf);

BMF_API AVFrame* bmf_af_to_avframe(const bmf_AudioFrame af);

#ifdef __cplusplus
} //extern "C"
#endif

#endif // ENGINE_WRAPPER_CUSTOM_FFMPEG_H
