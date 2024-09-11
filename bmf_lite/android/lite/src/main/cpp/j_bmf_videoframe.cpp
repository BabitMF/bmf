#include <jni.h>
#include <memory>
#include <string>
#include <vector>
#include "bmf_jni_helper.h"
#include "bmf_lite.h"
#include "algorithm/bmf_algorithm.h"
#include "error_code.h"
#include "log.h"

jlong createVideoFrame(JNIEnv *env, jobject instance) {
    bmf_lite::VideoFrame *videoFrame = new bmf_lite::VideoFrame();
    //    LOGI("jni processVideoFrame frame video_frame_ptr createVideoFrame =
    //    %d,",(long)(reinterpret_cast<jlong>(videoFrame)));
    if (videoFrame != NULL) {
        return reinterpret_cast<jlong>(videoFrame);
    }
    return (jlong)bmf_lite::BMF_LITE_StsNoMem;
}

jlong createTextureVideoFrame(JNIEnv *env, jobject instance, jint texture_id,
                              jint width, jint height) {
    int tex_width = (int)width;
    int tex_height = (int)height;
    bmf_lite::HWDeviceType device_type = bmf_lite::kHWDeviceTypeEGLCtx;
    std::shared_ptr<bmf_lite::HWDeviceContext> device_context_new;
    bmf_lite::HWDeviceContextManager::getCurrentHwDeviceContext(
        device_type, device_context_new);
    bmf_lite::HardwareDataInfo data_info = {
        bmf_lite::MemoryType::kOpenGLTexture2d, bmf_lite::GLES_TEXTURE_RGBA, 0};
    std::shared_ptr<bmf_lite::VideoBuffer> video_buffer;
    int tex_id = (int)(texture_id);
    bmf_lite::VideoBufferManager::createTextureVideoBufferFromExistingData(
        (void *)tex_id, tex_width, tex_height, &data_info, device_context_new,
        NULL, video_buffer);
    bmf_lite::VideoFrame *videoFrame = new bmf_lite::VideoFrame(video_buffer);
    if (videoFrame != NULL) {
        return reinterpret_cast<jlong>(videoFrame);
    } else {
        return (jlong)bmf_lite::BMF_LITE_StsNoMem;
    }
}

void releaseVideoFrame(JNIEnv *env, jobject instance,
                       jlong native_video_frame_ptr) {
    bmf_lite::VideoFrame *video_frame_ptr =
        reinterpret_cast<bmf_lite::VideoFrame *>(native_video_frame_ptr);
    if (video_frame_ptr != NULL) {
        delete video_frame_ptr;
        video_frame_ptr = nullptr;
    }
}

jint getTextureId(JNIEnv *env, jobject instance, jlong native_video_frame_ptr) {
    bmf_lite::VideoFrame *video_frame_ptr =
        reinterpret_cast<bmf_lite::VideoFrame *>(native_video_frame_ptr);
    if (video_frame_ptr == NULL) {
        return bmf_lite::BMF_LITE_NullPtr;
    }
    if (video_frame_ptr->buffer() == NULL) {
        //        LOGI("This is an log message from JNI!
        //        video_frame_ptr->buffer() == NULL");
        return bmf_lite::BMF_LITE_NullPtr;
    }

    long output_texture = (long)(video_frame_ptr->buffer()->data());
    int output_tex_id = int(output_texture);
    return (jint)output_tex_id;
}

jint getWidth(JNIEnv *env, jobject instance, jlong native_video_frame_ptr) {
    bmf_lite::VideoFrame *video_frame_ptr =
        reinterpret_cast<bmf_lite::VideoFrame *>(native_video_frame_ptr);
    if (video_frame_ptr == NULL) {
        return bmf_lite::BMF_LITE_NullPtr;
    }

    int width = (long)(video_frame_ptr->buffer()->width());
    return (jint)width;
}

jint getHeight(JNIEnv *env, jobject instance, jlong native_video_frame_ptr) {
    bmf_lite::VideoFrame *video_frame_ptr =
        reinterpret_cast<bmf_lite::VideoFrame *>(native_video_frame_ptr);
    if (video_frame_ptr == NULL) {
        return bmf_lite::BMF_LITE_NullPtr;
    }

    int height = (long)(video_frame_ptr->buffer()->height());
    return (jint)height;
}

jint setPixelFormat(JNIEnv *env, jobject instance, jlong native_video_frame_ptr,
                    jint pixFormat) {
    bmf_lite::VideoFrame *video_frame_ptr =
        reinterpret_cast<bmf_lite::VideoFrame *>(native_video_frame_ptr);
    if (video_frame_ptr == NULL) {
        return bmf_lite::BMF_LITE_NullPtr;
    }
    return 0;
}

static const JNINativeMethod gMethods[] = {
    {"nativeCreateVideoFrame", "()J", (void *)createVideoFrame},
    {"nativeCreateTextureVideoFrame", "(III)J",
     (void *)createTextureVideoFrame},
    {"nativeGetTextureId", "(J)I", (void *)getTextureId},
    {"nativeGetWidth", "(J)I", (void *)getWidth},
    {"nativeGetHeight", "(J)I", (void *)getHeight},
    {"nativeSetPixelFormat", "(JI)I", (void *)setPixelFormat},
    {"nativeReleaseVideoFrame", "(J)V", (void *)releaseVideoFrame},
};

int register_native_bmf_lite_video_frame(JNIEnv *env, const char *classPath) {
    return jniBmfRegisterNativeMethods(env, classPath, gMethods,
                                       NELEM(gMethods));
}
