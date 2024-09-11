
#include <assert.h>
#include <jni.h>
#include <stdexcept>
#include <string>
#include <vector>
#include <jni.h>
#include "log.h"

#ifdef BMF_LITE_ENABLE_ALGORITHM
static const char *BMF_LITE_ALGORITHM = "com/bmf/lite/AlgorithmInterface";
int register_native_bmf_lite_algorithm(JNIEnv *env, const char *classPath);
#endif

#ifdef BMF_LITE_ENABLE_PARAM
static const char *BMF_LITE_PARAM = "com/bmf/lite/Param";
int register_native_bmf_lite_param(JNIEnv *env, const char *classPath);
#endif

#ifdef BMF_LITE_ENABLE_VIDEO_FRAME
static const char *BMF_LITE_VIDEO_FRAME = "com/bmf/lite/VideoFrame";
int register_native_bmf_lite_video_frame(JNIEnv *env, const char *classPath);
#endif

static void UnregisterNativeMethods(JNIEnv *env, const char *className) {
    jclass clazz = env->FindClass(className);
    if (clazz == NULL) {
        return;
    }
    if (env != NULL) {
        env->UnregisterNatives(clazz);
    }
}

extern "C" jint JNI_OnLoad(JavaVM *vm, void *res) {
    JNIEnv *env = NULL;
    if (vm->GetEnv((void **)&env, JNI_VERSION_1_6) != JNI_OK) {
        return -1;
    }
    assert(env != NULL);
#ifdef BMF_LITE_ENABLE_ALGORITHM
    if (register_native_bmf_lite_algorithm(env, BMF_LITE_ALGORITHM) != 0) {
        return -1;
    }
#endif

#ifdef BMF_LITE_ENABLE_VIDEO_FRAME
    if (register_native_bmf_lite_param(env, BMF_LITE_PARAM) != 0) {
        return -1;
    }
#endif

#ifdef BMF_LITE_ENABLE_VIDEO_FRAME
    if (register_native_bmf_lite_video_frame(env, BMF_LITE_VIDEO_FRAME) != 0) {
        return -1;
    }
#endif
    return JNI_VERSION_1_6;
}

extern "C" void JNI_OnUnload(JavaVM *jvm, void *p) {
    JNIEnv *env = NULL;
    if (jvm->GetEnv((void **)(&env), JNI_VERSION_1_6) != JNI_OK) {
        return;
    }
#ifdef BMF_LITE_ENABLE_ALGORITHM
    UnregisterNativeMethods(env, BMF_LITE_ALGORITHM);
#endif
#ifdef BMF_LITE_ENABLE_VIDEO_FRAME
    UnregisterNativeMethods(env, BMF_LITE_VIDEO_FRAME);
#endif

#ifdef BMF_LITE_ENABLE_VIDEO_FRAME
    UnregisterNativeMethods(env, BMF_LITE_VIDEO_FRAME);
#endif
}
