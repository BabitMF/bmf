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

#include <stdio.h>
#include <stdlib.h>
#include <jni.h>
#include <dlfcn.h>
#include <asm-generic/siginfo.h>

typedef void (*JniInvocation_ctor_t)(void *);
typedef void (*JniInvocation_dtor_t)(void *);
typedef void (*JniInvocation_Init_t)(void *, const char *);
typedef int (*JNI_CreateJavaVM_t)(JavaVM **p_vm, JNIEnv **p_env, void *vm_args);
typedef jint (*registerNatives_t)(JNIEnv *env, jclass clazz);

int init_jvm(JavaVM **p_vm, JNIEnv **p_env);

#endif