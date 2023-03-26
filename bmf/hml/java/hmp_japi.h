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
#pragma once

#include <stdexcept>
#include <string>
#include <vector>
#include <jni.h>

#define JNI_PROTECT(...)                                        \
    try{                                                        \
        __VA_ARGS__                                             \
    } catch(const jni::JniException &e){                        \
    } catch(std::exception &e) {                                \
        auto exc_cls = env->FindClass("java/lang/Exception");   \
        env->ThrowNew(exc_cls, e.what());                       \
    }

namespace jni{

class JniException : public std::exception {
    public:
    static void checkException(JNIEnv *env)
    {
        if (env->ExceptionCheck()) {
            env->ExceptionDescribe();
            env->ExceptionClear();
            throw JniException("JNI ExceptionCheck found exception.");
        }
    }

  JniException(const std::string &message) : _message(message) {}
  ~JniException() noexcept override {}
  const char *what() const noexcept override { return _message.c_str(); }

 private:
  std::string _message;
};


// array helpers

template<typename T, typename U>
std::vector<T> fromJArray(JNIEnv *env, U arr);

template<>
std::vector<int64_t> fromJArray<int64_t, jlongArray>(JNIEnv *env, jlongArray arr)
{
    auto ptr = env->GetLongArrayElements(arr, nullptr);
    auto len = env->GetArrayLength(arr);
    return std::vector<int64_t>(ptr, ptr+len);
}

template<>
std::vector<int> fromJArray<int, jintArray>(JNIEnv *env, jintArray arr)
{
    auto ptr = env->GetIntArrayElements(arr, nullptr);
    auto len = env->GetArrayLength(arr);
    return std::vector<int>(ptr, ptr+len);
}


template<typename T, typename U>
U toJArray(JNIEnv *env, const std::vector<T>& arr);

template<>
jlongArray toJArray<int64_t, jlongArray>(JNIEnv *env, const std::vector<int64_t>& arr)
{
    auto output = env->NewLongArray(arr.size());
    env->SetLongArrayRegion(output, 0, arr.size(), &arr[0]);
    return output;
}

template<>
jintArray toJArray<int, jintArray>(JNIEnv *env, const std::vector<int>& arr)
{
    auto output = env->NewIntArray(arr.size());
    env->SetIntArrayRegion(output, 0, arr.size(), &arr[0]);
    return output;
}


// string & jstring

std::string fromJString(JNIEnv *env, jstring jstr, const std::string &defaultValue="", bool deleteLocalRef=false) {
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


jstring toJString(JNIEnv *env, const std::string &str)
{
    return env->NewStringUTF(str.c_str());
}

// ptr helpers
template<typename T, typename ...Args>
static int64_t makePtr(Args&&...args){
    return reinterpret_cast<int64_t>(new T(std::forward<Args>(args)...));
}

template<typename T>
static void freePtr(int64_t ptr)
{
    if(ptr){
        delete (T*)ptr;
    }
}

template<typename T>
static T* ptr(int64_t p, bool check=true)
{
    if(check && p == 0){
        throw std::runtime_error(std::string("Null pointer detected"));
    }

    return reinterpret_cast<T*>(p);
}



} //namespace
