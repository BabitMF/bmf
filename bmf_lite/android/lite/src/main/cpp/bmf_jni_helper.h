
#ifndef _BMF_JNIHELP_H_
#define _BMF_JNIHELP_H_

#include <jni.h>
#include <string>
#include <vector>
#ifndef NELEM
#define NELEM(x) ((int)(sizeof(x) / sizeof((x)[0])))
#endif

#ifdef __cplusplus
extern "C" {
#endif

int jniBmfRegisterNativeMethods(JNIEnv *env, const char *className,
                                const JNINativeMethod *gMethods,
                                int numMethods);

#ifdef __cplusplus
}
#endif

#endif
