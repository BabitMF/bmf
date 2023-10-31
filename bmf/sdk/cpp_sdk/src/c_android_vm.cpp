/*
 * Copyright 2023 Babit Authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifdef BMF_USE_MEDIACODEC
#include "../include/c_android_vm.h"

#include "log.h"

const int JniAllocErr = -1;
const int NoCreateJavaVMSymbolErr = -2;
const int NoRegisterNativesSymbolErr = -3;
const int CreateJavaFuncErr = -4;
const int RegisterNativesErr = -5;

int init_jvm(JavaVM **p_vm, JNIEnv **p_env) {

    JavaVMOption opt[2];
    opt[0].optionString = "-Xplugin:libopenjdkjvmti.so";

    JavaVMInitArgs args;
    args.version = JNI_VERSION_1_6;
    args.options = opt;
    args.nOptions =
        1; // Uptick this to 5, it will pass in the no_sig_chain option
    args.ignoreUnrecognized = JNI_FALSE;

    void *libandroid_runtime_dso = dlopen("libnativehelper.so", RTLD_NOW);
    void *libandroid_runtime_dso2 = dlopen("libandroid_runtime.so", RTLD_NOW);
    void *jni = NULL;
    JniInvocation_ctor_t JniInvocation_ctor;
    JniInvocation_dtor_t JniInvocation_dtor;
    JniInvocation_Init_t JniInvocation_Init;
    JniInvocation_ctor = (JniInvocation_ctor_t)dlsym(
        libandroid_runtime_dso, "_ZN17JniInvocationImplC1Ev");
    JniInvocation_dtor = (JniInvocation_dtor_t)dlsym(
        libandroid_runtime_dso, "_ZN17JniInvocationImplD1Ev");
    JniInvocation_Init = (JniInvocation_Init_t)dlsym(
        libandroid_runtime_dso, "_ZN17JniInvocationImpl4InitEPKc");
    if (JniInvocation_ctor && JniInvocation_dtor && JniInvocation_Init) {
        jni = calloc(1, 256);
        if (!jni) {
            BMFLOG(BMF_ERROR) << "cannot alloc Memory for JNI Invocation!";
            return JniAllocErr;
        }
        JniInvocation_ctor(jni);
        JniInvocation_Init(jni, NULL);
    }
    JNI_CreateJavaVM_t JNI_CreateJavaVM;
    JNI_CreateJavaVM =
        (JNI_CreateJavaVM_t)dlsym(libandroid_runtime_dso, "JNI_CreateJavaVM");
    if (!JNI_CreateJavaVM) {
        BMFLOG(BMF_ERROR) << "cannot Find CreateJavaVM Symbol!";
        return NoCreateJavaVMSymbolErr;
    }

    registerNatives_t registerNatives;
    registerNatives = (registerNatives_t)dlsym(
        libandroid_runtime_dso2,
        "Java_com_android_internal_util_WithFramework_registerNatives");
    if (!registerNatives) {
        // Attempt non-legacy version
        registerNatives = (registerNatives_t)dlsym(libandroid_runtime_dso2,
                                                   "registerFrameworkNatives");
        if (!registerNatives) {
            BMFLOG(BMF_ERROR) << "cannot Find RegisterNatives Symbol!";
            return NoRegisterNativesSymbolErr;
        }
    }

    if (JNI_CreateJavaVM(&(*p_vm), &(*p_env), &args)) {
        BMFLOG(BMF_ERROR) << "Call JNI_CreateJavaVM Func Err!";
        return CreateJavaFuncErr;
    }

    if (registerNatives(*p_env, 0)) {
        BMFLOG(BMF_ERROR) << "Call registerNatives Func Err!";
        return RegisterNativesErr;
    }

    return 0;
}

#endif