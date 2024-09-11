#include "bmf_lite.h"
#include "algorithm/bmf_algorithm.h"
#include "bmf_jni_helper.h"
#include "log.h"
#include "error_code.h"
jlong createAlgorithm(JNIEnv *env, jobject instance) {
    bmf_lite::IAlgorithm *algorithm_ptr =
        bmf_lite::AlgorithmFactory::createAlgorithmInterface();
    if (algorithm_ptr != nullptr) {
        return reinterpret_cast<jlong>(algorithm_ptr);
    }
    return (jlong)bmf_lite::BMF_LITE_StsNoMem;
}

jint setAlgorithmParam(JNIEnv *env, jobject instance, jlong native_alg_ptr,
                       jlong native_param_ptr) {
    bmf_lite::IAlgorithm *algorithm_ptr =
        reinterpret_cast<bmf_lite::IAlgorithm *>(native_alg_ptr);
    if (algorithm_ptr == nullptr) {
        return (jint)bmf_lite::BMF_LITE_NullPtr;
    }
    bmf_lite::Param *param_ptr =
        reinterpret_cast<bmf_lite::Param *>(native_param_ptr);
    if (param_ptr == nullptr) {
        return (jint)bmf_lite::BMF_LITE_NullPtr;
    }
    int ret = algorithm_ptr->setParam(*param_ptr);
    //    LOGI("This is an log message from JNI setAlgorithmParam ret =
    //    %d",ret);
    return jint(ret);
}

jint processVideoFrame(JNIEnv *env, jobject instance,
                       jlong native_algorithm_ptr, jlong native_video_frame_ptr,
                       jlong native_process_param_ptr) {
    bmf_lite::IAlgorithm *algorithm_ptr =
        reinterpret_cast<bmf_lite::IAlgorithm *>(native_algorithm_ptr);
    if (algorithm_ptr == nullptr) {
        return (jint)bmf_lite::BMF_LITE_NullPtr;
    }
    bmf_lite::Param *param_ptr =
        reinterpret_cast<bmf_lite::Param *>(native_process_param_ptr);
    if (param_ptr == nullptr) {
        return (jint)bmf_lite::BMF_LITE_NullPtr;
    }
    bmf_lite::VideoFrame *video_frame_ptr =
        reinterpret_cast<bmf_lite::VideoFrame *>(native_video_frame_ptr);
    if (video_frame_ptr == nullptr) {
        return (jint)bmf_lite::BMF_LITE_NullPtr;
    }

    int ret = algorithm_ptr->processVideoFrame(*video_frame_ptr, *param_ptr);
    return jint(ret);
}

jobject getVideoFrameOutput(JNIEnv *env, jobject instance,
                            jlong native_algorithm_ptr) {
    jclass videoframe_output_class =
        env->FindClass("com/bmf/lite/VideoFrameOutput");
    if (videoframe_output_class == nullptr) {
        return nullptr;
    }
    jmethodID constructor =
        env->GetMethodID(videoframe_output_class, "<init>", "(IJJ)V");
    if (constructor == nullptr) {
        return nullptr;
    }

    bmf_lite::IAlgorithm *algorithm_ptr =
        reinterpret_cast<bmf_lite::IAlgorithm *>(native_algorithm_ptr);
    if (algorithm_ptr == nullptr) {
        return nullptr;
    }
    bmf_lite::Param *param_ptr = new bmf_lite::Param();
    if (param_ptr == nullptr) {
        return nullptr;
    }
    bmf_lite::VideoFrame *video_frame_ptr =
        new bmf_lite::VideoFrame(); // reinterpret_cast<bmf_lite::BmfVideoFrame
                                    // *>(video_frame_datas[0]);
    if (video_frame_ptr == nullptr) {
        return nullptr;
    }
    int ret = algorithm_ptr->getVideoFrameOutput(*video_frame_ptr, *param_ptr);

    jobject java_obj =
        env->NewObject(videoframe_output_class, constructor, (jint)ret,
                       (jlong) reinterpret_cast<jlong>(video_frame_ptr),
                       (jlong) reinterpret_cast<jlong>(param_ptr));

    //    LOGI("jni processVideoFrame frame texid =
    //    %d,",(long)(video_frame_ptr->buffer()->data()));
    return java_obj;
}

jobject getProcessProperty(JNIEnv *env, jobject instance,
                           jlong native_algorithm_ptr) {
    jclass property_class = env->FindClass("com/bmf/lite/PropertyParam");
    if (property_class == nullptr) {
        return nullptr;
    }
    jmethodID constructor = env->GetMethodID(property_class, "<init>", "(IJ)V");
    if (constructor == nullptr) {
        return nullptr;
    }

    bmf_lite::IAlgorithm *algorithm_ptr =
        reinterpret_cast<bmf_lite::IAlgorithm *>(native_algorithm_ptr);
    if (algorithm_ptr == nullptr) {
        return nullptr;
    }
    bmf_lite::Param *param_ptr = new bmf_lite::Param();
    if (param_ptr == nullptr) {
        return nullptr;
    }

    int ret = algorithm_ptr->getProcessProperty(*param_ptr);

    jobject java_obj =
        env->NewObject(property_class, constructor, (jint)ret,
                       (jlong) reinterpret_cast<jlong>(param_ptr));
    return java_obj;
}

jint setInputProperty(JNIEnv *env, jobject instance, jlong native_algorithm_ptr,
                      jlong native_param_ptr) {
    bmf_lite::IAlgorithm *algorithm_ptr =
        reinterpret_cast<bmf_lite::IAlgorithm *>(native_algorithm_ptr);
    if (algorithm_ptr == nullptr) {
        return (jint)bmf_lite::BMF_LITE_NullPtr;
    }
    bmf_lite::Param *param_ptr =
        reinterpret_cast<bmf_lite::Param *>(native_param_ptr);
    if (param_ptr == nullptr) {
        return (jint)bmf_lite::BMF_LITE_NullPtr;
    }
    int ret = algorithm_ptr->setInputProperty(*param_ptr);
    return jint(ret);

    return (jint)bmf_lite::BMF_LITE_StsOk;
}

jobject getOutputProperty(JNIEnv *env, jobject instance,
                          jlong native_algorithm_ptr) {
    jclass property_class = env->FindClass("com/bmf/lite/PropertyParam");
    if (property_class == nullptr) {
        return nullptr;
    }
    jmethodID constructor = env->GetMethodID(property_class, "<init>", "(IJ)V");
    if (constructor == nullptr) {
        return nullptr;
    }

    bmf_lite::IAlgorithm *algorithm_ptr =
        reinterpret_cast<bmf_lite::IAlgorithm *>(native_algorithm_ptr);
    if (algorithm_ptr == nullptr) {
        return nullptr;
    }
    bmf_lite::Param *param_ptr = new bmf_lite::Param();
    if (param_ptr == nullptr) {
        return nullptr;
    }

    int ret = algorithm_ptr->getOutputProperty(*param_ptr);

    jobject java_obj =
        env->NewObject(property_class, constructor, (jint)ret,
                       (jlong) reinterpret_cast<jlong>(param_ptr));
    return java_obj;
}

void releaseAlgorithm(JNIEnv *env, jobject instance,
                      jlong native_algorithm_ptr) {
    bmf_lite::IAlgorithm *algorithm =
        reinterpret_cast<bmf_lite::IAlgorithm *>(native_algorithm_ptr);
    if (algorithm != nullptr) {
        delete algorithm;
        algorithm = nullptr;
    }
}

static const JNINativeMethod gMethods[] = {
    {"nativeCreateAlgorithm", "()J", (void *)createAlgorithm},
    {"nativeReleaseAlgorithm", "(J)V", (void *)releaseAlgorithm},
    {"nativeSetAlgorithmParam", "(JJ)I", (void *)setAlgorithmParam},
    {"nativeProcessVideoFrame", "(JJJ)I", (void *)processVideoFrame},
    {"nativeGetVideoFrameOutput", "(J)Lcom/bmf/lite/VideoFrameOutput;",
     (void *)getVideoFrameOutput},
    {"nativeGetProcessProperty", "(J)Lcom/bmf/lite/PropertyParam;",
     (void *)getProcessProperty},
    {"nativeSetInputProperty", "(JJ)I", (void *)setInputProperty},
    {"nativeGetOutputProperty", "(J)Lcom/bmf/lite/PropertyParam;",
     (void *)getOutputProperty},
};
int register_native_bmf_lite_algorithm(JNIEnv *env, const char *classPath) {
    return jniBmfRegisterNativeMethods(env, classPath, gMethods,
                                       NELEM(gMethods));
}