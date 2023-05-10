#ifdef BMF_USE_MEDIACODEC

#include <stdio.h>
#include <stdlib.h>
#include <jni.h>
#include <dlfcn.h>
#include <asm-generic/siginfo.h>

typedef void (*JniInvocation_ctor_t)(void *);
typedef void (*JniInvocation_dtor_t)(void *);
typedef void (*JniInvocation_Init_t)(void *, const char *);
typedef int(*JNI_CreateJavaVM_t)(JavaVM **p_vm, JNIEnv **p_env, void *vm_args);
typedef jint(*registerNatives_t)(JNIEnv *env, jclass clazz);

int init_jvm(JavaVM **p_vm, JNIEnv **p_env);

#endif