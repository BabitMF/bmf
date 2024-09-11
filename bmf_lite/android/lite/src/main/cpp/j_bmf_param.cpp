#include <jni.h>
#include <memory>
#include <string>
#include <vector>
#include "bmf_jni_helper.h"
#include "bmf_lite.h"
#include "algorithm/bmf_algorithm.h"
#include "error_code.h"

std::string jString2StdString(JNIEnv *env, jstring jstr,
                              const std::string &defaultValue = "",
                              bool deleteLocalRef = false) {
    const char *chars = env->GetStringUTFChars(jstr, NULL);
    if (chars == nullptr) {
        return defaultValue;
    }
    std::string ret = chars;
    env->ReleaseStringUTFChars(jstr, chars);
    if (deleteLocalRef) {
        env->DeleteLocalRef(jstr);
    }
    return ret;
}

jlong createAlgorithmParam(JNIEnv *env, jobject instance) {
    bmf_lite::Param *param_ptr = new bmf_lite::Param();
    if (param_ptr != NULL) {
        return reinterpret_cast<jlong>(param_ptr);
    }
    return (jlong)bmf_lite::BMF_LITE_StsNoMem;
}

jboolean hasKey(JNIEnv *env, jobject instance, jlong native_param_ptr,
                jstring key) {
    bmf_lite::Param *param_ptr =
        reinterpret_cast<bmf_lite::Param *>(native_param_ptr);
    if (param_ptr == NULL) {
        return (jboolean) false;
    }
    return param_ptr->has_key(jString2StdString(env, key));
}

jint eraseKey(JNIEnv *env, jobject instance, jlong native_param_ptr,
              jstring key) {
    bmf_lite::Param *param_ptr =
        reinterpret_cast<bmf_lite::Param *>(native_param_ptr);
    if (param_ptr == NULL) {
        return (jint)bmf_lite::BMF_LITE_NullPtr;
    }
    return (jint)(param_ptr->erase(jString2StdString(env, key)));
}

void releaseAlgorithmParam(JNIEnv *env, jobject instance,
                           jlong native_param_ptr) {
    bmf_lite::Param *param_ptr =
        reinterpret_cast<bmf_lite::Param *>(native_param_ptr);
    if (param_ptr != NULL) {
        delete param_ptr;
    }
}

jint setInt(JNIEnv *env, jobject instance, jlong native_param_ptr, jstring key,
            jint value) {
    bmf_lite::Param *param_ptr =
        reinterpret_cast<bmf_lite::Param *>(native_param_ptr);
    if (param_ptr == NULL) {
        return (jint)bmf_lite::BMF_LITE_NullPtr;
    }
    return param_ptr->setInt(jString2StdString(env, key), value);
}

jint setLong(JNIEnv *env, jobject instance, jlong native_param_ptr, jstring key,
             jlong value) {
    bmf_lite::Param *param_ptr =
        reinterpret_cast<bmf_lite::Param *>(native_param_ptr);
    if (param_ptr == NULL) {
        return (jint)bmf_lite::BMF_LITE_NullPtr;
    }
    return param_ptr->setLong(jString2StdString(env, key), value);
}

jint setFloat(JNIEnv *env, jobject instance, jlong native_param_ptr,
              jstring key, jfloat value) {
    bmf_lite::Param *param_ptr =
        reinterpret_cast<bmf_lite::Param *>(native_param_ptr);
    if (param_ptr == NULL) {
        return (jint)bmf_lite::BMF_LITE_NullPtr;
    }
    return param_ptr->setFloat(jString2StdString(env, key), value);
}

jint setDouble(JNIEnv *env, jobject instance, jlong native_param_ptr,
               jstring key, jdouble value) {
    bmf_lite::Param *param_ptr =
        reinterpret_cast<bmf_lite::Param *>(native_param_ptr);
    if (param_ptr == NULL) {
        return (jint)bmf_lite::BMF_LITE_NullPtr;
    }
    return param_ptr->setDouble(jString2StdString(env, key), value);
}

jint setString(JNIEnv *env, jobject instance, jlong native_param_ptr,
               jstring key, jstring value) {
    bmf_lite::Param *param_ptr =
        reinterpret_cast<bmf_lite::Param *>(native_param_ptr);
    if (param_ptr == NULL) {
        return (jint)bmf_lite::BMF_LITE_NullPtr;
    }
    return param_ptr->setString(jString2StdString(env, key),
                                jString2StdString(env, value));
}

jint setIntList(JNIEnv *env, jobject instance, jlong native_param_ptr,
                jstring key, jintArray values) {
    bmf_lite::Param *param_ptr =
        reinterpret_cast<bmf_lite::Param *>(native_param_ptr);
    if (param_ptr == NULL) {
        return (jint)bmf_lite::BMF_LITE_NullPtr;
    }
    jint *int_data = env->GetIntArrayElements(values, nullptr);
    if (int_data == nullptr) {
        return (jint)bmf_lite::BMF_LITE_JNI;
    }
    int length = env->GetArrayLength(values);
    std::vector<int> int_data_vec(int_data, int_data + length);
    int result =
        param_ptr->setIntList(jString2StdString(env, key), int_data_vec);
    env->ReleaseIntArrayElements(values, int_data, 0);
    return result;
}

jint setFloatList(JNIEnv *env, jobject instance, jlong native_param_ptr,
                  jstring key, jfloatArray values) {
    bmf_lite::Param *param_ptr =
        reinterpret_cast<bmf_lite::Param *>(native_param_ptr);
    if (param_ptr == NULL) {
        return (jint)bmf_lite::BMF_LITE_NullPtr;
    }
    jfloat *float_data = env->GetFloatArrayElements(values, nullptr);
    if (float_data == nullptr) {
        return (jint)bmf_lite::BMF_LITE_JNI;
    }
    int length = env->GetArrayLength(values);
    std::vector<float> float_data_vec(float_data, float_data + length);
    int result =
        param_ptr->setFloatList(jString2StdString(env, key), float_data_vec);
    env->ReleaseFloatArrayElements(values, float_data, 0);
    return result;
}

jint setDoubleList(JNIEnv *env, jobject instance, jlong native_param_ptr,
                   jstring key, jdoubleArray values) {
    bmf_lite::Param *param_ptr =
        reinterpret_cast<bmf_lite::Param *>(native_param_ptr);
    if (param_ptr == NULL) {
        return (jint)bmf_lite::BMF_LITE_NullPtr;
    }
    jdouble *double_data = env->GetDoubleArrayElements(values, nullptr);
    if (double_data == nullptr) {
        return (jint)bmf_lite::BMF_LITE_JNI;
    }
    int length = env->GetArrayLength(values);
    std::vector<double> double_data_vec(double_data, double_data + length);
    int result =
        param_ptr->setDoubleList(jString2StdString(env, key), double_data_vec);
    env->ReleaseDoubleArrayElements(values, double_data, 0);
    return result;
}

// jint setStringList(JNIEnv *env, jobject instance,
//                     jlong native_param_ptr,
//                     jstring key,jstring[] values)
//{
//     return 0;
// }

jint getInt(JNIEnv *env, jobject instance, jlong native_param_ptr,
            jstring key) {
    bmf_lite::Param *param_ptr =
        reinterpret_cast<bmf_lite::Param *>(native_param_ptr);
    if (param_ptr == NULL) {
        return (jint)0;
    }
    int result = 0;
    int status = param_ptr->getInt(jString2StdString(env, key), result);
    if (status != 0) {
        return (jint)0;
    }
    return (jint)result;
}

jlong getLong(JNIEnv *env, jobject instance, jlong native_param_ptr,
              jstring key) {
    bmf_lite::Param *param_ptr =
        reinterpret_cast<bmf_lite::Param *>(native_param_ptr);
    if (param_ptr == NULL) {
        return (jlong)0;
    }
    int64_t result = 0;
    int status = param_ptr->getLong(jString2StdString(env, key), result);
    if (status != 0) {
        return (jlong)0;
    }
    return (jlong)result;
}

jfloat getFloat(JNIEnv *env, jobject instance, jlong native_param_ptr,
                jstring key) {
    bmf_lite::Param *param_ptr =
        reinterpret_cast<bmf_lite::Param *>(native_param_ptr);
    if (param_ptr == NULL) {
        return (jfloat)0;
    }
    float result = 0;
    int status = param_ptr->getFloat(jString2StdString(env, key), result);
    if (status != 0) {
        return (jfloat)0;
    }
    return (jfloat)result;
}

jdouble getDouble(JNIEnv *env, jobject instance, jlong native_param_ptr,
                  jstring key) {
    bmf_lite::Param *param_ptr =
        reinterpret_cast<bmf_lite::Param *>(native_param_ptr);
    if (param_ptr == NULL) {
        return (jdouble)0;
    }
    double result = 0;
    int status = param_ptr->getDouble(jString2StdString(env, key), result);
    if (status != 0) {
        return (jdouble)0;
    }
    return (jdouble)result;
}

jstring getString(JNIEnv *env, jobject instance, jlong native_param_ptr,
                  jstring key) {
    bmf_lite::Param *param_ptr =
        reinterpret_cast<bmf_lite::Param *>(native_param_ptr);
    if (param_ptr == NULL) {
        std::string tmp_str = "";
        return env->NewStringUTF(tmp_str.c_str());
    }
    std::string result;
    int status = param_ptr->getString(jString2StdString(env, key), result);
    if (status != 0) {
        std::string tmp_str = "";
        return env->NewStringUTF(tmp_str.c_str());
    }
    return env->NewStringUTF(result.c_str());
}

jintArray getIntList(JNIEnv *env, jobject instance, jlong native_param_ptr,
                     jstring key) {
    bmf_lite::Param *param_ptr =
        reinterpret_cast<bmf_lite::Param *>(native_param_ptr);
    if (param_ptr == NULL) {
        return nullptr;
    }
    std::vector<int> int_list;
    param_ptr->getIntList(jString2StdString(env, key), int_list);
    int len = int_list.size();
    jintArray int_array = env->NewIntArray(len);
    if (int_array == NULL) {
        return nullptr;
    }
    jint buf[len];
    for (int i = 0; i < len; i++) {
        buf[i] = int_list[i];
    }
    env->SetIntArrayRegion(int_array, 0, len, buf);
    return int_array;
}

// jlongArray getLongList(JNIEnv *env, jobject instance,
//                             jlong native_param_ptr,
//                             jstring key)
//{
//     bmf_lite::Param* param_ptr = reinterpret_cast<bmf_lite::Param
//     *>(native_param_ptr); if (param_ptr == NULL)
//     {
//         return nullptr;
//     }
//     return param_ptr->getLongList(jString2StdString(env, key));
// }

jfloatArray getFloatList(JNIEnv *env, jobject instance, jlong native_param_ptr,
                         jstring key) {
    bmf_lite::Param *param_ptr =
        reinterpret_cast<bmf_lite::Param *>(native_param_ptr);
    if (param_ptr == NULL) {
        return nullptr;
    }
    std::vector<float> float_list;
    param_ptr->getFloatList(jString2StdString(env, key), float_list);
    int len = float_list.size();
    jfloatArray float_array = env->NewFloatArray(len);
    if (float_array == NULL) {
        return nullptr;
    }
    jfloat buf[len];
    for (int i = 0; i < len; i++) {
        buf[i] = float_list[i];
    }
    env->SetFloatArrayRegion(float_array, 0, len, buf);
    return float_array;
}

jdoubleArray getDoubleList(JNIEnv *env, jobject instance,
                           jlong native_param_ptr, jstring key) {
    bmf_lite::Param *param_ptr =
        reinterpret_cast<bmf_lite::Param *>(native_param_ptr);
    if (param_ptr == NULL) {
        return nullptr;
    }
    std::vector<double> double_list;
    param_ptr->getDoubleList(jString2StdString(env, key), double_list);
    int len = double_list.size();
    jdoubleArray double_array = env->NewDoubleArray(len);
    if (double_array == NULL) {
        return nullptr;
    }
    jdouble buf[len];
    for (int i = 0; i < len; i++) {
        buf[i] = double_list[i];
    }
    env->SetDoubleArrayRegion(double_array, 0, len, buf);
    return double_array;
}

static const JNINativeMethod gMethods[] = {
    {"nativeCreateAlgorithmParam", "()J", (void *)createAlgorithmParam},
    {"nativeReleaseAlgorithmParam", "(J)V", (void *)releaseAlgorithmParam},
    {"nativeHasKey", "(JLjava/lang/String;)Z", (void *)hasKey},
    {"nativeEraseKey", "(JLjava/lang/String;)I", (void *)eraseKey},
    {"nativeSetInt", "(JLjava/lang/String;I)I", (void *)setInt},
    {"nativeSetLong", "(JLjava/lang/String;J)I", (void *)setLong},
    {"nativeSetFloat", "(JLjava/lang/String;F)I", (void *)setFloat},
    {"nativeSetDouble", "(JLjava/lang/String;D)I", (void *)setDouble},
    {"nativeSetString", "(JLjava/lang/String;Ljava/lang/String;)I",
     (void *)setString},
    {"nativeSetIntList", "(JLjava/lang/String;[I)I", (void *)setIntList},
    //        {"nativeSetLongList", "(JLjava/lang/String;[J)I", (void
    //        *)setLongList},
    {"nativeSetFloatList", "(JLjava/lang/String;[F)I", (void *)setFloatList},
    {"nativeSetDoubleList", "(JLjava/lang/String;[D)I", (void *)setDoubleList},
    //    {"nativeSetStringList", "(JLjava/lang/String;[Ljava/lang/String;)I",
    //    (void *)setStringList},
    {"nativeGetInt", "(JLjava/lang/String;)I", (void *)getInt},
    {"nativeGetLong", "(JLjava/lang/String;)J", (void *)getLong},
    {"nativeGetFloat", "(JLjava/lang/String;)F", (void *)getFloat},
    {"nativeGetDouble", "(JLjava/lang/String;)D", (void *)getDouble},
    {"nativeGetString", "(JLjava/lang/String;)Ljava/lang/String;",
     (void *)getString},
    {"nativeGetIntList", "(JLjava/lang/String;)[I", (void *)getIntList},
    //        {"nativeGetLongList", "(JLjava/lang/String;)[J", (void
    //        *)getLongList},
    {"nativeGetFloatList", "(JLjava/lang/String;)[F", (void *)getFloatList},
    {"nativeGetDoubleList", "(JLjava/lang/String;)[D", (void *)getDoubleList},
};
//    {"nativeGetStringList", "(JLjava/lang/String;)[Ljava/lang/String;", (void
//    *)getStringList}

int register_native_bmf_lite_param(JNIEnv *env, const char *classPath) {
    return jniBmfRegisterNativeMethods(env, classPath, gMethods,
                                       NELEM(gMethods));
}
