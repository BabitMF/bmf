#include <pthread.h>
#include <stdlib.h>
#include <assert.h>
#include "bmf_jni_helper.h"

int jniBmfRegisterNativeMethods(JNIEnv *env, const char *className,
                                const JNINativeMethod *gMethods,
                                int numMethods) {

    jclass clazz = env->FindClass(className);
    if (clazz == nullptr) {
        return -1;
    }

    int result = 0;
    if (env->RegisterNatives(clazz, gMethods, numMethods) < 0) {
        result = -1;
    }
    env->DeleteLocalRef(clazz);
    return result;
}