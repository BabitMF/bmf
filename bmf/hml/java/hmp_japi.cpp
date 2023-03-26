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

#include <android/bitmap.h>
#include "com_bytedance_hmp_Api.h"
#include "hmp_japi.h"
#include <hmp/tensor.h>
#include <hmp/imgproc/image.h>
#include <hmp/core/stream.h>


using namespace hmp;



////////////////////// Scalar //////////////////////////
/*
 * Class:     com_bytedance_hmp_Api
 * Method:    scalar
 * Signature: (D)J
 */
JNIEXPORT jlong JNICALL Java_com_bytedance_hmp_Api_scalar__D
  (JNIEnv *env, jclass, jdouble value)
{
    JNI_PROTECT(
        return jni::makePtr<Scalar>(value);
    )

    return 0;
}

/*
 * Class:     com_bytedance_hmp_Api
 * Method:    scalar
 * Signature: (J)J
 */
JNIEXPORT jlong JNICALL Java_com_bytedance_hmp_Api_scalar__J
  (JNIEnv *env, jclass, jlong value)
{
    JNI_PROTECT(
        return jni::makePtr<Scalar>(value);
    )

    return 0;
}

/*
 * Class:     com_bytedance_hmp_Api
 * Method:    scalar
 * Signature: (Z)J
 */
JNIEXPORT jlong JNICALL Java_com_bytedance_hmp_Api_scalar__Z
  (JNIEnv *env, jclass, jboolean value)
{
    JNI_PROTECT(
        return jni::makePtr<Scalar>((bool)value);
    )

    return 0;
}

/*
 * Class:     com_bytedance_hmp_Api
 * Method:    scalar_free
 * Signature: (J)V
 */
JNIEXPORT void JNICALL Java_com_bytedance_hmp_Api_scalar_1free
  (JNIEnv *env, jclass, jlong ptr)
{
    JNI_PROTECT(
        jni::freePtr<Scalar>(ptr);
    )
}


////////////////////// Device //////////////////////////////////////

/*
 * Class:     com_bytedance_hmp_Api
 * Method:    device_count
 * Signature: (I)I
 */
JNIEXPORT jint JNICALL Java_com_bytedance_hmp_Api_device_1count
  (JNIEnv *env, jclass, jint device_type)
{
    JNI_PROTECT(
        return device_count((DeviceType)device_type);
    )

    return 0;
}

/*
 * Class:     com_bytedance_hmp_Api
 * Method:    device_make
 * Signature: (Ljava/lang/String;)J
 */
JNIEXPORT jlong JNICALL Java_com_bytedance_hmp_Api_device_1make__Ljava_lang_String_2
  (JNIEnv *env, jclass, jstring dstr)
{
    JNI_PROTECT(
        auto str = jni::fromJString(env, dstr);
        if(str.size()){
            return jni::makePtr<Device>(str);
        }
        else{
            return jni::makePtr<Device>();
        }
    )
    return 0;
}


/*
 * Class:     com_bytedance_hmp_Api
 * Method:    device_make
 * Signature: (II)J
 */
JNIEXPORT jlong JNICALL Java_com_bytedance_hmp_Api_device_1make__II
  (JNIEnv *env, jclass, jint device_type, jint index)
{
    JNI_PROTECT(
        return jni::makePtr<Device>((DeviceType)device_type, index);
    )
    return 0;
}

/*
 * Class:     com_bytedance_hmp_Api
 * Method:    device_free
 * Signature: (J)V
 */
JNIEXPORT void JNICALL Java_com_bytedance_hmp_Api_device_1free
  (JNIEnv *env, jclass, jlong ptr)
{
    JNI_PROTECT(
        jni::freePtr<Scalar>(ptr);
    )
}

/*
 * Class:     com_bytedance_hmp_Api
 * Method:    device_type
 * Signature: (J)I
 */
JNIEXPORT jint JNICALL Java_com_bytedance_hmp_Api_device_1type
  (JNIEnv *env, jclass, jlong ptr)
{
    return (jint)jni::ptr<Device>(ptr)->type();

}

/*
 * Class:     com_bytedance_hmp_Api
 * Method:    device_index
 * Signature: (J)I
 */
JNIEXPORT jint JNICALL Java_com_bytedance_hmp_Api_device_1index
  (JNIEnv *, jclass, jlong ptr)
{
    return (jint)jni::ptr<Device>(ptr)->index();
}


/*
 * Class:     com_bytedance_hmp_Api
 * Method:    device_stringfy
 * Signature: (J)Ljava/lang/String;
 */
JNIEXPORT jstring JNICALL Java_com_bytedance_hmp_Api_device_1stringfy
  (JNIEnv *env, jclass, jlong ptr)
{
    JNI_PROTECT(
        auto str = stringfy(*jni::ptr<Device>(ptr));
        return jni::toJString(env, str);
    )
    return jni::toJString(env, "");
}


/*
 * Class:     com_bytedance_hmp_Api
 * Method:    device_guard_make
 * Signature: (J)J
 */
JNIEXPORT jlong JNICALL Java_com_bytedance_hmp_Api_device_1guard_1make
  (JNIEnv *env, jclass, jlong device)
{
    JNI_PROTECT(
        return jni::makePtr<DeviceGuard>(*jni::ptr<Device>(device));
    )
    return 0;
}

/*
 * Class:     com_bytedance_hmp_Api
 * Method:    device_guard_free
 * Signature: (J)V
 */
JNIEXPORT void JNICALL Java_com_bytedance_hmp_Api_device_1guard_1free
  (JNIEnv *env, jclass, jlong ptr)
{
    JNI_PROTECT(
      jni::freePtr<DeviceGuard>(ptr);
    )
}


///////////////////// Stream & StreamGuard ////////////////
/*
 * Class:     com_bytedance_hmp_Api
 * Method:    stream_make
 * Signature: (IJ)J
 */
JNIEXPORT jlong JNICALL Java_com_bytedance_hmp_Api_stream_1make
  (JNIEnv *env, jclass, jint device_type, jlong flags)
{
    JNI_PROTECT(
        return jni::makePtr<Stream>(create_stream((DeviceType)device_type, flags));
    )
    return 0;
}

/*
 * Class:     com_bytedance_hmp_Api
 * Method:    stream_free
 * Signature: (J)V
 */
JNIEXPORT void JNICALL Java_com_bytedance_hmp_Api_stream_1free
  (JNIEnv *env, jclass, jlong ptr)
{
    JNI_PROTECT(
        jni::freePtr<Stream>(ptr);
    )
}

/*
 * Class:     com_bytedance_hmp_Api
 * Method:    stream_query
 * Signature: (J)Z
 */
JNIEXPORT jboolean JNICALL Java_com_bytedance_hmp_Api_stream_1query
  (JNIEnv *env, jclass, jlong ptr)
{
    JNI_PROTECT(
        return jni::ptr<Stream>(ptr)->query();
    )
    return false;
}

/*
 * Class:     com_bytedance_hmp_Api
 * Method:    stream_synchronize
 * Signature: (J)V
 */
JNIEXPORT void JNICALL Java_com_bytedance_hmp_Api_stream_1synchronize
  (JNIEnv *env, jclass, jlong ptr)
{
    JNI_PROTECT(
        jni::ptr<Stream>(ptr)->synchronize();
    )
}

/*
 * Class:     com_bytedance_hmp_Api
 * Method:    stream_handle
 * Signature: (J)J
 */
JNIEXPORT jlong JNICALL Java_com_bytedance_hmp_Api_stream_1handle
  (JNIEnv *env, jclass, jlong ptr)
{
    JNI_PROTECT(
        return jni::ptr<Stream>(ptr)->handle();
    )
    return 0;
}

/*
 * Class:     com_bytedance_hmp_Api
 * Method:    stream_device_type
 * Signature: (J)I
 */
JNIEXPORT jint JNICALL Java_com_bytedance_hmp_Api_stream_1device_1type
  (JNIEnv *env, jclass, jlong ptr)
{
    JNI_PROTECT(
        return (jint)jni::ptr<Stream>(ptr)->device().type();
    )
    return 0;
}

/*
 * Class:     com_bytedance_hmp_Api
 * Method:    stream_device_index
 * Signature: (J)I
 */
JNIEXPORT jint JNICALL Java_com_bytedance_hmp_Api_stream_1device_1index
  (JNIEnv *env, jclass, jlong ptr)
{
    JNI_PROTECT(
        return (jint)jni::ptr<Stream>(ptr)->device().index();
    )
    return 0;
}

/*
 * Class:     com_bytedance_hmp_Api
 * Method:    stream_set_current
 * Signature: (J)V
 */
JNIEXPORT void JNICALL Java_com_bytedance_hmp_Api_stream_1set_1current
  (JNIEnv *env, jclass, jlong ptr)
{
    JNI_PROTECT(
        set_current_stream(*jni::ptr<Stream>(ptr));
    )
}

/*
 * Class:     com_bytedance_hmp_Api
 * Method:    stream_current
 * Signature: (I)J
 */
JNIEXPORT jlong JNICALL Java_com_bytedance_hmp_Api_stream_1current
  (JNIEnv *env, jclass, jint device_type)
{
    JNI_PROTECT(
        return jni::makePtr<Stream>(current_stream((DeviceType)device_type).value());
    )
    return 0;
}

/*
 * Class:     com_bytedance_hmp_Api
 * Method:    stream_guard_create
 * Signature: (J)J
 */
JNIEXPORT jlong JNICALL Java_com_bytedance_hmp_Api_stream_1guard_1create
  (JNIEnv *env, jclass, jlong stream)
{
    JNI_PROTECT(
        return jni::makePtr<StreamGuard>(*jni::ptr<Stream>(stream));
    )
    return 0;
}

/*
 * Class:     com_bytedance_hmp_Api
 * Method:    stream_guard_free
 * Signature: (J)V
 */
JNIEXPORT void JNICALL Java_com_bytedance_hmp_Api_stream_1guard_1free
  (JNIEnv *env, jclass, jlong ptr)
{
    JNI_PROTECT(
        jni::freePtr<StreamGuard>(ptr);
    )
}


/////////////////////// Tensor /////////////////////////////
/*
 * Class:     com_bytedance_hmp_Api
 * Method:    tensor_empty
 * Signature: ([JILjava/lang/String;Z)J
 */
JNIEXPORT jlong JNICALL Java_com_bytedance_hmp_Api_tensor_1empty
  (JNIEnv *env, jclass, jlongArray shape, jint dtype, jstring device, jboolean pinned_memory)
{
    JNI_PROTECT(
        auto vshape = jni::fromJArray<int64_t, jlongArray>(env, shape);
        return jni::makePtr<Tensor>(empty(vshape, TensorOptions(jni::fromJString(env, device))
                                           .dtype((ScalarType)dtype)
                                           .pinned_memory(pinned_memory)));
    )
    return 0;
}

/*
 * Class:     com_bytedance_hmp_Api
 * Method:    tensor_arange
 * Signature: (JJJILjava/lang/String;Z)J
 */
JNIEXPORT jlong JNICALL Java_com_bytedance_hmp_Api_tensor_1arange
  (JNIEnv *env, jclass, jlong start, jlong end, jlong step, jint dtype, jstring device, jboolean pinned_memory)
{
    JNI_PROTECT(
        return jni::makePtr<Tensor>(arange(start, end, step, 
                                    TensorOptions(jni::fromJString(env, device))
                                           .dtype((ScalarType)dtype)
                                           .pinned_memory(pinned_memory)));
    )
    return 0;
}

/*
 * Class:     com_bytedance_hmp_Api
 * Method:    tensor_free
 * Signature: (J)V
 */
JNIEXPORT void JNICALL Java_com_bytedance_hmp_Api_tensor_1free
  (JNIEnv *env, jclass, jlong ptr)
{
    JNI_PROTECT(
        jni::freePtr<Tensor>(ptr);
    )
}

/*
 * Class:     com_bytedance_hmp_Api
 * Method:    tensor_stringfy
 * Signature: (J)Ljava/lang/String;
 */
JNIEXPORT jstring JNICALL Java_com_bytedance_hmp_Api_tensor_1stringfy
  (JNIEnv *env, jclass, jlong ptr)
{
    JNI_PROTECT(
        auto str = stringfy(*jni::ptr<Tensor>(ptr));
        return jni::toJString(env, str);
    )

    return jni::toJString(env, "");
}

/*
 * Class:     com_bytedance_hmp_Api
 * Method:    tensor_fill
 * Signature: (JJ)V
 */
JNIEXPORT void JNICALL Java_com_bytedance_hmp_Api_tensor_1fill
  (JNIEnv *env, jclass, jlong ptr, jlong scalar)
{
    JNI_PROTECT(
        jni::ptr<Tensor>(ptr)->fill_(*jni::ptr<Scalar>(scalar));
    )
}

/*
 * Class:     com_bytedance_hmp_Api
 * Method:    tensor_defined
 * Signature: (J)Z
 */
JNIEXPORT jboolean JNICALL Java_com_bytedance_hmp_Api_tensor_1defined
  (JNIEnv *env, jclass, jlong ptr)
{
    JNI_PROTECT(
        return jni::ptr<Tensor>(ptr)->defined();
    )
    return false;
}

/*
 * Class:     com_bytedance_hmp_Api
 * Method:    tensor_dim
 * Signature: (J)J
 */
JNIEXPORT jlong JNICALL Java_com_bytedance_hmp_Api_tensor_1dim
  (JNIEnv *env, jclass, jlong ptr)
{
    JNI_PROTECT(
        return jni::ptr<Tensor>(ptr)->dim();
    )
    return 0;
}

/*
 * Class:     com_bytedance_hmp_Api
 * Method:    tensor_size
 * Signature: (JJ)J
 */
JNIEXPORT jlong JNICALL Java_com_bytedance_hmp_Api_tensor_1size
  (JNIEnv *env, jclass, jlong ptr, jlong dim)
{
    JNI_PROTECT(
        return jni::ptr<Tensor>(ptr)->size(dim);
    )
    return 0;
}

/*
 * Class:     com_bytedance_hmp_Api
 * Method:    tensor_stride
 * Signature: (JJ)J
 */
JNIEXPORT jlong JNICALL Java_com_bytedance_hmp_Api_tensor_1stride
  (JNIEnv *env, jclass, jlong ptr, jlong dim)
{
    JNI_PROTECT(
        return jni::ptr<Tensor>(ptr)->stride(dim);
    )
    return 0;
}

/*
 * Class:     com_bytedance_hmp_Api
 * Method:    tensor_nitems
 * Signature: (J)J
 */
JNIEXPORT jlong JNICALL Java_com_bytedance_hmp_Api_tensor_1nitems
  (JNIEnv *env, jclass, jlong ptr)
{
    JNI_PROTECT(
        return jni::ptr<Tensor>(ptr)->nitems();
    )
    return 0;
}


/*
 * Class:     com_bytedance_hmp_Api
 * Method:    tensor_from_file
 * Signature: (Ljava/lang/String;IJJ)J
 */
JNIEXPORT jlong JNICALL Java_com_bytedance_hmp_Api_tensor_1from_1file
  (JNIEnv *env, jclass, jstring fn, jint dtype, jlong count, jlong offset)
{
  JNI_PROTECT(
    auto sfn = jni::fromJString(env, fn);
    return jni::makePtr<Tensor>(
            hmp::fromfile(sfn, (ScalarType)dtype, count, offset));
  )

  return 0;
}

/*
 * Class:     com_bytedance_hmp_Api
 * Method:    tensor_to_file
 * Signature: (JLjava/lang/String;)V
 */
JNIEXPORT void JNICALL Java_com_bytedance_hmp_Api_tensor_1to_1file
  (JNIEnv *env, jclass, jlong data, jstring fn)
{
  JNI_PROTECT(
    auto sfn = jni::fromJString(env, fn);
    hmp::tofile(*jni::ptr<Tensor>(data), sfn);
  )
}


/*
 * Class:     com_bytedance_hmp_Api
 * Method:    tensor_itemsize
 * Signature: (J)J
 */
JNIEXPORT jlong JNICALL Java_com_bytedance_hmp_Api_tensor_1itemsize
  (JNIEnv *env, jclass, jlong ptr)
{
    JNI_PROTECT(
        return jni::ptr<Tensor>(ptr)->itemsize();
    )
    return 0;
}

/*
 * Class:     com_bytedance_hmp_Api
 * Method:    tensor_nbytes
 * Signature: (J)J
 */
JNIEXPORT jlong JNICALL Java_com_bytedance_hmp_Api_tensor_1nbytes
  (JNIEnv *env, jclass, jlong ptr)
{
    JNI_PROTECT(
        return jni::ptr<Tensor>(ptr)->nbytes();
    )
    return 0;
}

/*
 * Class:     com_bytedance_hmp_Api
 * Method:    tensor_dtype
 * Signature: (J)I
 */
JNIEXPORT jint JNICALL Java_com_bytedance_hmp_Api_tensor_1dtype
  (JNIEnv *env, jclass, jlong ptr)
{
    JNI_PROTECT(
        return (jint)jni::ptr<Tensor>(ptr)->dtype();
    )
    return 0;
}

/*
 * Class:     com_bytedance_hmp_Api
 * Method:    tensor_is_contiguous
 * Signature: (J)Z
 */
JNIEXPORT jboolean JNICALL Java_com_bytedance_hmp_Api_tensor_1is_1contiguous
  (JNIEnv *env, jclass, jlong ptr)
{
    JNI_PROTECT(
        return jni::ptr<Tensor>(ptr)->is_contiguous();
    )
    return 0;
}

/*
 * Class:     com_bytedance_hmp_Api
 * Method:    tensor_device_type
 * Signature: (J)I
 */
JNIEXPORT jint JNICALL Java_com_bytedance_hmp_Api_tensor_1device_1type
  (JNIEnv *env, jclass, jlong ptr)
{
    JNI_PROTECT(
        return (jint)jni::ptr<Tensor>(ptr)->device_type();
    )
    return 0;
}

/*
 * Class:     com_bytedance_hmp_Api
 * Method:    tensor_device_index
 * Signature: (J)I
 */
JNIEXPORT jint JNICALL Java_com_bytedance_hmp_Api_tensor_1device_1index
  (JNIEnv *env, jclass, jlong ptr)
{
    JNI_PROTECT(
        return jni::ptr<Tensor>(ptr)->device_index();
    )
    return 0;
}

/*
 * Class:     com_bytedance_hmp_Api
 * Method:    tensor_data_ptr
 * Signature: (J)J
 */
JNIEXPORT jlong JNICALL Java_com_bytedance_hmp_Api_tensor_1data_1ptr
  (JNIEnv *env, jclass, jlong ptr)
{
    JNI_PROTECT(
      return (jlong)jni::ptr<Tensor>(ptr)->unsafe_data();
    )
    return 0;
}


/*
 * Class:     com_bytedance_hmp_Api
 * Method:    tensor_clone
 * Signature: (J)J
 */
JNIEXPORT jlong JNICALL Java_com_bytedance_hmp_Api_tensor_1clone
  (JNIEnv *env, jclass, jlong ptr)
{
    JNI_PROTECT(
        return jni::makePtr<Tensor>(jni::ptr<Tensor>(ptr)->clone());
    )
    return 0;
}

/*
 * Class:     com_bytedance_hmp_Api
 * Method:    tensor_alias
 * Signature: (J)J
 */
JNIEXPORT jlong JNICALL Java_com_bytedance_hmp_Api_tensor_1alias
  (JNIEnv *env, jclass, jlong ptr)
{
    JNI_PROTECT(
        return jni::makePtr<Tensor>(jni::ptr<Tensor>(ptr)->alias());
    )
    return 0;
}


/*
 * Class:     com_bytedance_hmp_Api
 * Method:    tensor_view
 * Signature: (J[J)J
 */
JNIEXPORT jlong JNICALL Java_com_bytedance_hmp_Api_tensor_1view
  (JNIEnv *env, jclass, jlong ptr, jlongArray shape)
{
    JNI_PROTECT(
        auto vshape = jni::fromJArray<int64_t>(env, shape);
        return jni::makePtr<Tensor>(jni::ptr<Tensor>(ptr)->view(vshape));
    )
    return 0;
}

/*
 * Class:     com_bytedance_hmp_Api
 * Method:    tensor_reshape
 * Signature: (J[J)J
 */
JNIEXPORT jlong JNICALL Java_com_bytedance_hmp_Api_tensor_1reshape
  (JNIEnv *env, jclass, jlong ptr, jlongArray shape)
{
    JNI_PROTECT(
        auto vshape = jni::fromJArray<int64_t>(env, shape);
        return jni::makePtr<Tensor>(jni::ptr<Tensor>(ptr)->reshape(vshape));
    )
    return 0;
}

/*
 * Class:     com_bytedance_hmp_Api
 * Method:    tensor_slice
 * Signature: (JJJJJ)J
 */
JNIEXPORT jlong JNICALL Java_com_bytedance_hmp_Api_tensor_1slice
  (JNIEnv *env, jclass, jlong ptr, jlong dim, jlong start, jlong end, jlong step)
{
    JNI_PROTECT(
        return jni::makePtr<Tensor>(jni::ptr<Tensor>(ptr)->slice(dim, start, end, step));
    )
    return 0;
}

/*
 * Class:     com_bytedance_hmp_Api
 * Method:    tensor_select
 * Signature: (JJJ)J
 */
JNIEXPORT jlong JNICALL Java_com_bytedance_hmp_Api_tensor_1select
  (JNIEnv *env, jclass, jlong ptr, jlong dim, jlong index)
{
    JNI_PROTECT(
        return jni::makePtr<Tensor>(jni::ptr<Tensor>(ptr)->select(dim, index));
    )
    return 0;
}

/*
 * Class:     com_bytedance_hmp_Api
 * Method:    tensor_permute
 * Signature: (J[J)J
 */
JNIEXPORT jlong JNICALL Java_com_bytedance_hmp_Api_tensor_1permute
  (JNIEnv *env, jclass, jlong ptr, jlongArray dims)
{
    JNI_PROTECT(
        auto vdims = jni::fromJArray<int64_t>(env, dims);
        return jni::makePtr<Tensor>(jni::ptr<Tensor>(ptr)->permute(vdims));
    )
    return 0;
}

/*
 * Class:     com_bytedance_hmp_Api
 * Method:    tensor_squeeze
 * Signature: (JJ)J
 */
JNIEXPORT jlong JNICALL Java_com_bytedance_hmp_Api_tensor_1squeeze
  (JNIEnv *env, jclass, jlong ptr, jlong dim)
{
    JNI_PROTECT(
        return jni::makePtr<Tensor>(jni::ptr<Tensor>(ptr)->squeeze(dim));
    )
    return 0;
}

/*
 * Class:     com_bytedance_hmp_Api
 * Method:    tensor_unsqueeze
 * Signature: (JJ)J
 */
JNIEXPORT jlong JNICALL Java_com_bytedance_hmp_Api_tensor_1unsqueeze
  (JNIEnv *env, jclass, jlong ptr, jlong dim)
{
    JNI_PROTECT(
        return jni::makePtr<Tensor>(jni::ptr<Tensor>(ptr)->unsqueeze(dim));
    )
    return 0;
}

/*
 * Class:     com_bytedance_hmp_Api
 * Method:    tensor_to_device
 * Signature: (JLjava/lang/String;Z)J
 */
JNIEXPORT jlong JNICALL Java_com_bytedance_hmp_Api_tensor_1to_1device
  (JNIEnv *env, jclass, jlong ptr, jstring device, jboolean non_blocking)
{
    JNI_PROTECT(
        return jni::makePtr<Tensor>(jni::ptr<Tensor>(ptr)->to(
            Device(jni::fromJString(env, device)), non_blocking));
    )
    return 0;
}

/*
 * Class:     com_bytedance_hmp_Api
 * Method:    tensor_to_dtype
 * Signature: (JI)J
 */
JNIEXPORT jlong JNICALL Java_com_bytedance_hmp_Api_tensor_1to_1dtype
  (JNIEnv *env, jclass, jlong ptr, jint dtype)
{
    JNI_PROTECT(
        return jni::makePtr<Tensor>(jni::ptr<Tensor>(ptr)->to(
            ScalarType(dtype)));
    )
    return 0;
}

/*
 * Class:     com_bytedance_hmp_Api
 * Method:    tensor_copy_from
 * Signature: (JJ)V
 */
JNIEXPORT void JNICALL Java_com_bytedance_hmp_Api_tensor_1copy_1from
  (JNIEnv *env, jclass, jlong ptr, jlong from)
{
    JNI_PROTECT(
        jni::ptr<Tensor>(ptr)->copy_(*jni::ptr<Tensor>(from));
    )
}


/*
 * Class:     com_bytedance_hmp_Api
 * Method:    color_model_make
 * Signature: (IIII)J
 */
JNIEXPORT jlong JNICALL Java_com_bytedance_hmp_Api_color_1model_1make
  (JNIEnv *env, jclass, jint cs, jint cr, jint cp, jint ctc)
{
    JNI_PROTECT(
        return jni::makePtr<ColorModel>((ColorSpace)cs, 
                                         (ColorRange)cr, 
                                         (ColorPrimaries)cp,
                                         (ColorTransferCharacteristic)ctc);
    )

    return 0;
}

/*
 * Class:     com_bytedance_hmp_Api
 * Method:    color_model_free
 * Signature: (J)V
 */
JNIEXPORT void JNICALL Java_com_bytedance_hmp_Api_color_1model_1free
  (JNIEnv *env, jclass, jlong ptr)
{
    JNI_PROTECT(
        jni::freePtr<ColorModel>(ptr);
    )
}

/*
 * Class:     com_bytedance_hmp_Api
 * Method:    color_model_space
 * Signature: (J)I
 */
JNIEXPORT jint JNICALL Java_com_bytedance_hmp_Api_color_1model_1space
  (JNIEnv *env, jclass, jlong ptr)
{
    JNI_PROTECT(
        return jni::ptr<ColorModel>(ptr)->space();
    )
    return 0;
}

/*
 * Class:     com_bytedance_hmp_Api
 * Method:    color_model_range
 * Signature: (J)I
 */
JNIEXPORT jint JNICALL Java_com_bytedance_hmp_Api_color_1model_1range
  (JNIEnv *env, jclass, jlong ptr)
{
    JNI_PROTECT(
        return jni::ptr<ColorModel>(ptr)->range();
    )
    return 0;
}

/*
 * Class:     com_bytedance_hmp_Api
 * Method:    color_model_primaries
 * Signature: (J)I
 */
JNIEXPORT jint JNICALL Java_com_bytedance_hmp_Api_color_1model_1primaries
  (JNIEnv *env, jclass, jlong ptr)
{
    JNI_PROTECT(
        return jni::ptr<ColorModel>(ptr)->primaries();
    )
    return 0;
}

/*
 * Class:     com_bytedance_hmp_Api
 * Method:    color_model_ctc
 * Signature: (J)I
 */
JNIEXPORT jint JNICALL Java_com_bytedance_hmp_Api_color_1model_1ctc
  (JNIEnv *env, jclass, jlong ptr)
{
    JNI_PROTECT(
        return jni::ptr<ColorModel>(ptr)->transfer_characteristic();
    )
    return 0;
}

/*
 * Class:     com_bytedance_hmp_Api
 * Method:    pixel_info_make
 * Signature: (IJ)J
 */
JNIEXPORT jlong JNICALL Java_com_bytedance_hmp_Api_pixel_1info_1make__IJ
  (JNIEnv *env, jclass, jint format, jlong cm)
{
    JNI_PROTECT(
        return jni::makePtr<PixelInfo>((PixelFormat)format, *jni::ptr<ColorModel>(cm));
    )
    return 0;
}

/*
 * Class:     com_bytedance_hmp_Api
 * Method:    pixel_info_make
 * Signature: (III)J
 */
JNIEXPORT jlong JNICALL Java_com_bytedance_hmp_Api_pixel_1info_1make__III
  (JNIEnv *env, jclass, jint format, jint cs, jint cr)
{
    JNI_PROTECT(
        return jni::makePtr<PixelInfo>((PixelFormat)format,
                                       (ColorSpace)cs,
                                       (ColorRange)cr);
    )
    return 0;
}

/*
 * Class:     com_bytedance_hmp_Api
 * Method:    pixel_info_free
 * Signature: (J)V
 */
JNIEXPORT void JNICALL Java_com_bytedance_hmp_Api_pixel_1info_1free
  (JNIEnv *env, jclass, jlong ptr)
{
    JNI_PROTECT(
        jni::freePtr<PixelInfo>(ptr);
    )
}

/*
 * Class:     com_bytedance_hmp_Api
 * Method:    pixel_info_format
 * Signature: (J)I
 */
JNIEXPORT jint JNICALL Java_com_bytedance_hmp_Api_pixel_1info_1format
  (JNIEnv *env, jclass, jlong ptr)
{
    JNI_PROTECT(
        return (jint)jni::ptr<PixelInfo>(ptr)->format();
    )
    return 0;
}

/*
 * Class:     com_bytedance_hmp_Api
 * Method:    pixel_info_space
 * Signature: (J)I
 */
JNIEXPORT jint JNICALL Java_com_bytedance_hmp_Api_pixel_1info_1space
  (JNIEnv *env, jclass, jlong ptr)
{
    JNI_PROTECT(
        return (jint)jni::ptr<PixelInfo>(ptr)->space();
    )
    return 0;
}

/*
 * Class:     com_bytedance_hmp_Api
 * Method:    pixel_info_range
 * Signature: (J)I
 */
JNIEXPORT jint JNICALL Java_com_bytedance_hmp_Api_pixel_1info_1range
  (JNIEnv *env, jclass, jlong ptr)
{
    JNI_PROTECT(
        return (jint)jni::ptr<PixelInfo>(ptr)->range();
    )
    return 0;
}

/*
 * Class:     com_bytedance_hmp_Api
 * Method:    pixel_info_primaries
 * Signature: (J)I
 */
JNIEXPORT jint JNICALL Java_com_bytedance_hmp_Api_pixel_1info_1primaries
  (JNIEnv *env, jclass, jlong ptr)
{
    JNI_PROTECT(
        return (jint)jni::ptr<PixelInfo>(ptr)->primaries();
    )
    return 0;
}

/*
 * Class:     com_bytedance_hmp_Api
 * Method:    pixel_info_ctc
 * Signature: (J)I
 */
JNIEXPORT jint JNICALL Java_com_bytedance_hmp_Api_pixel_1info_1ctc
  (JNIEnv *env, jclass, jlong ptr)
{
    JNI_PROTECT(
        return (jint)jni::ptr<PixelInfo>(ptr)->transfer_characteristic();
    )
    return 0;
}

/*
 * Class:     com_bytedance_hmp_Api
 * Method:    pixel_info_infer_space
 * Signature: (J)I
 */
JNIEXPORT jint JNICALL Java_com_bytedance_hmp_Api_pixel_1info_1infer_1space
  (JNIEnv *env, jclass, jlong ptr)
{
    JNI_PROTECT(
        return (jint)jni::ptr<PixelInfo>(ptr)->infer_space();
    )
    return 0;
}

/*
 * Class:     com_bytedance_hmp_Api
 * Method:    pixel_info_color_model
 * Signature: (J)J
 */
JNIEXPORT jlong JNICALL Java_com_bytedance_hmp_Api_pixel_1info_1color_1model
  (JNIEnv *env, jclass, jlong ptr)
{
    JNI_PROTECT(
        return (jlong)(&jni::ptr<PixelInfo>(ptr)->color_model());
    )
    return 0;
}

/*
 * Class:     com_bytedance_hmp_Api
 * Method:    pixel_info_is_rgbx
 * Signature: (J)Z
 */
JNIEXPORT jboolean JNICALL Java_com_bytedance_hmp_Api_pixel_1info_1is_1rgbx
  (JNIEnv *env, jclass, jlong ptr)
{
    JNI_PROTECT(
        return jni::ptr<PixelInfo>(ptr)->is_rgbx();
    )
    return 0;
}

/*
 * Class:     com_bytedance_hmp_Api
 * Method:    pixel_info_stringfy
 * Signature: (J)Ljava/lang/String;
 */
JNIEXPORT jstring JNICALL Java_com_bytedance_hmp_Api_pixel_1info_1stringfy
  (JNIEnv *env, jclass, jlong ptr)
{
    JNI_PROTECT(
        auto str = stringfy(*jni::ptr<PixelInfo>(ptr));
        return jni::toJString(env, str);
    )
    return jni::toJString(env, "");
}

/*
 * Class:     com_bytedance_hmp_Api
 * Method:    pixel_format_desc_make
 * Signature: (I)J
 */
JNIEXPORT jlong JNICALL Java_com_bytedance_hmp_Api_pixel_1format_1desc_1make
  (JNIEnv *env, jclass obj, jint format)
{
    JNI_PROTECT(
        return jni::makePtr<PixelFormatDesc>(format);
    )
    return 0;
}

/*
 * Class:     com_bytedance_hmp_Api
 * Method:    pixel_format_desc_free
 * Signature: (J)V
 */
JNIEXPORT void JNICALL Java_com_bytedance_hmp_Api_pixel_1format_1desc_1free
  (JNIEnv *env, jclass obj, jlong ptr)
{
    JNI_PROTECT(
        jni::freePtr<PixelFormatDesc>(ptr);
    )
}

/*
 * Class:     com_bytedance_hmp_Api
 * Method:    pixel_format_desc_nplanes
 * Signature: (J)I
 */
JNIEXPORT jint JNICALL Java_com_bytedance_hmp_Api_pixel_1format_1desc_1nplanes
  (JNIEnv *env, jclass obj, jlong ptr)
{
    JNI_PROTECT(
        return jni::ptr<PixelFormatDesc>(ptr)->nplanes();
    )
    return 0;
}

/*
 * Class:     com_bytedance_hmp_Api
 * Method:    pixel_format_desc_dtype
 * Signature: (J)I
 */
JNIEXPORT jint JNICALL Java_com_bytedance_hmp_Api_pixel_1format_1desc_1dtype
  (JNIEnv *env, jclass obj, jlong ptr)
{
    JNI_PROTECT(
        return (int)jni::ptr<PixelFormatDesc>(ptr)->dtype();
    )
    return 0;
}

/*
 * Class:     com_bytedance_hmp_Api
 * Method:    pixel_format_desc_format
 * Signature: (J)I
 */
JNIEXPORT jint JNICALL Java_com_bytedance_hmp_Api_pixel_1format_1desc_1format
  (JNIEnv *env, jclass obj, jlong ptr)
{
    JNI_PROTECT(
        return jni::ptr<PixelFormatDesc>(ptr)->format();
    )
    return 0;
}

/*
 * Class:     com_bytedance_hmp_Api
 * Method:    pixel_format_desc_channels
 * Signature: (JI)I
 */
JNIEXPORT jint JNICALL Java_com_bytedance_hmp_Api_pixel_1format_1desc_1channels
  (JNIEnv *env, jclass obj, jlong ptr, jint plane)
{
    JNI_PROTECT(
        return jni::ptr<PixelFormatDesc>(ptr)->channels();
    )
    return 0;
}

/*
 * Class:     com_bytedance_hmp_Api
 * Method:    pixel_format_desc_infer_width
 * Signature: (JII)I
 */
JNIEXPORT jint JNICALL Java_com_bytedance_hmp_Api_pixel_1format_1desc_1infer_1width
  (JNIEnv *env, jclass obj, jlong ptr, jint width, jint plane)
{
    JNI_PROTECT(
        return jni::ptr<PixelFormatDesc>(ptr)->infer_width(width, plane);
    )
    return 0;
}

/*
 * Class:     com_bytedance_hmp_Api
 * Method:    pixel_format_desc_infer_height
 * Signature: (JII)I
 */
JNIEXPORT jint JNICALL Java_com_bytedance_hmp_Api_pixel_1format_1desc_1infer_1height
  (JNIEnv *env, jclass obj, jlong ptr, jint height, jint plane)
{
    JNI_PROTECT(
        return jni::ptr<PixelFormatDesc>(ptr)->infer_height(height, plane);
    )
    return 0;
}

/*
 * Class:     com_bytedance_hmp_Api
 * Method:    pixel_format_desc_infer_nitems
 * Signature: (JII)I
 */
JNIEXPORT jint JNICALL Java_com_bytedance_hmp_Api_pixel_1format_1desc_1infer_1nitems__JII
  (JNIEnv *env, jclass obj, jlong ptr, jint width, jint height)
{
    JNI_PROTECT(
        return jni::ptr<PixelFormatDesc>(ptr)->infer_nitems(width, height);
    )
    return 0;
}

/*
 * Class:     com_bytedance_hmp_Api
 * Method:    pixel_format_desc_infer_nitems
 * Signature: (JIII)I
 */
JNIEXPORT jint JNICALL Java_com_bytedance_hmp_Api_pixel_1format_1desc_1infer_1nitems__JIII
  (JNIEnv *env, jclass obj, jlong ptr, jint width, jint height, jint plane)
{
    JNI_PROTECT(
        return jni::ptr<PixelFormatDesc>(ptr)->infer_nitems(width, height, plane);
    )
    return 0;
}

/*
 * Class:     com_bytedance_hmp_Api
 * Method:    frame_make
 * Signature: (IIJLjava/lang/String;)J
 */
JNIEXPORT jlong JNICALL Java_com_bytedance_hmp_Api_frame_1make__IIJLjava_lang_String_2
  (JNIEnv *env, jclass, jint width, jint height, jlong pix_info, jstring device)
{
    JNI_PROTECT(
        auto dstr = jni::fromJString(env, device);
        return jni::makePtr<Frame>(width, height, *jni::ptr<PixelInfo>(pix_info), dstr);
    )
    return 0;
}


/*
 * Class:     com_bytedance_hmp_Api
 * Method:    frame_make
 * Signature: ([JJ)J
 */
JNIEXPORT jlong JNICALL Java_com_bytedance_hmp_Api_frame_1make___3JJ
  (JNIEnv *env, jclass, jlongArray data, jlong pix_info)
{
    JNI_PROTECT(
        TensorList vdata;
        for(auto p : jni::fromJArray<int64_t>(env, data)){
            vdata.push_back(*jni::ptr<Tensor>(p));
        }

        return jni::makePtr<Frame>(vdata, *jni::ptr<PixelInfo>(pix_info));
    )

    return 0;
}

/*
 * Class:     com_bytedance_hmp_Api
 * Method:    frame_make
 * Signature: ([JIIJ)J
 */
JNIEXPORT jlong JNICALL Java_com_bytedance_hmp_Api_frame_1make___3JIIJ
  (JNIEnv *env, jclass, jlongArray data, jint width, jint height, jlong pix_info)
{
    JNI_PROTECT(
        TensorList vdata;
        for(auto p : jni::fromJArray<int64_t>(env, data)){
            vdata.push_back(*jni::ptr<Tensor>(p));
        }

        return jni::makePtr<Frame>(vdata, width, height, *jni::ptr<PixelInfo>(pix_info));
    )

    return 0;
}

/*
 * Class:     com_bytedance_hmp_Api
 * Method:    frame_make
 * Signature: (Landroid/graphics/Bitmap;)J
 */
JNIEXPORT jlong JNICALL Java_com_bytedance_hmp_Api_frame_1make__Landroid_graphics_Bitmap_2
  (JNIEnv *env, jclass, jobject bitmap)
{
  JNI_PROTECT(
    AndroidBitmapInfo bitmapInfo;
    int ret = AndroidBitmap_getInfo(env, bitmap, &bitmapInfo);
    HMP_REQUIRE(ret >= 0, "Get Bitmap info failed, ret={}", ret);
    HMP_REQUIRE(bitmapInfo.format == ANDROID_BITMAP_FORMAT_RGBA_8888, 
               "Only RGBA_8888 format is supported");

    void *ptr;
    ret = AndroidBitmap_lockPixels(env, bitmap, &ptr);
    HMP_REQUIRE(ret >= 0, "Lock Bitmap pixels failed, ret={}", ret);
    auto data_ptr = DataPtr(ptr, 
                      [=](void*){ AndroidBitmap_unlockPixels(env, bitmap); },
                      Device("cpu"));
    SizeArray shape{bitmapInfo.height, bitmapInfo.width, 4};
    SizeArray strides{bitmapInfo.stride, 4, 1};
    Tensor data = from_buffer(std::move(data_ptr), kUInt8, shape, strides);
    return jni::makePtr<Frame>(TensorList{data}, PixelInfo(PF_RGBA32));
  )

  return 0;
}

/*
 * Class:     com_bytedance_hmp_Api
 * Method:    frame_free
 * Signature: (J)V
 */
JNIEXPORT void JNICALL Java_com_bytedance_hmp_Api_frame_1free
  (JNIEnv *env, jclass, jlong ptr)
{
    JNI_PROTECT(
        jni::freePtr<Frame>(ptr);
    )
}

/*
 * Class:     com_bytedance_hmp_Api
 * Method:    frame_defined
 * Signature: (J)Z
 */
JNIEXPORT jboolean JNICALL Java_com_bytedance_hmp_Api_frame_1defined
  (JNIEnv *env, jclass, jlong ptr)
{
    JNI_PROTECT(
        return jni::ptr<Frame>(ptr)->operator bool();
    )
    return false;
}

/*
 * Class:     com_bytedance_hmp_Api
 * Method:    frame_pix_info
 * Signature: (J)J
 */
JNIEXPORT jlong JNICALL Java_com_bytedance_hmp_Api_frame_1pix_1info
  (JNIEnv *env, jclass, jlong ptr)
{
    JNI_PROTECT(
        return (jlong)(&jni::ptr<Frame>(ptr)->pix_info());
    )
    return 0;
}

/*
 * Class:     com_bytedance_hmp_Api
 * Method:    frame_format
 * Signature: (J)I
 */
JNIEXPORT jint JNICALL Java_com_bytedance_hmp_Api_frame_1format
  (JNIEnv *env, jclass, jlong ptr)
{
    JNI_PROTECT(
        return (jint)jni::ptr<Frame>(ptr)->format();
    )
    return 0;
}

/*
 * Class:     com_bytedance_hmp_Api
 * Method:    frame_width
 * Signature: (J)I
 */
JNIEXPORT jint JNICALL Java_com_bytedance_hmp_Api_frame_1width
  (JNIEnv *env, jclass, jlong ptr)
{
    JNI_PROTECT(
        return jni::ptr<Frame>(ptr)->width();
    )
    return 0;
}

/*
 * Class:     com_bytedance_hmp_Api
 * Method:    frame_height
 * Signature: (J)I
 */
JNIEXPORT jint JNICALL Java_com_bytedance_hmp_Api_frame_1height
  (JNIEnv *env, jclass, jlong ptr)
{
    JNI_PROTECT(
        return jni::ptr<Frame>(ptr)->height();
    )
    return 0;
}

/*
 * Class:     com_bytedance_hmp_Api
 * Method:    frame_dtype
 * Signature: (J)I
 */
JNIEXPORT jint JNICALL Java_com_bytedance_hmp_Api_frame_1dtype
  (JNIEnv *env, jclass, jlong ptr)
{
    JNI_PROTECT(
        return (jint)jni::ptr<Frame>(ptr)->dtype();
    )
    return 0;

}

/*
 * Class:     com_bytedance_hmp_Api
 * Method:    frame_device_type
 * Signature: (J)I
 */
JNIEXPORT jint JNICALL Java_com_bytedance_hmp_Api_frame_1device_1type
  (JNIEnv *env, jclass, jlong ptr)
{
    JNI_PROTECT(
        return (jint)jni::ptr<Frame>(ptr)->device().type();
    )
    return 0;
}

/*
 * Class:     com_bytedance_hmp_Api
 * Method:    frame_device_index
 * Signature: (J)I
 */
JNIEXPORT jint JNICALL Java_com_bytedance_hmp_Api_frame_1device_1index
  (JNIEnv *env, jclass, jlong ptr)
{
    JNI_PROTECT(
        return (jint)jni::ptr<Frame>(ptr)->device().index();
    )
    return 0;
}


/*
 * Class:     com_bytedance_hmp_Api
 * Method:    frame_nplanes
 * Signature: (J)I
 */
JNIEXPORT jint JNICALL Java_com_bytedance_hmp_Api_frame_1nplanes
  (JNIEnv *env, jclass, jlong ptr)
{
    JNI_PROTECT(
        return jni::ptr<Frame>(ptr)->nplanes();
    )
    return 0;
}

/*
 * Class:     com_bytedance_hmp_Api
 * Method:    frame_plane
 * Signature: (JI)J
 */
JNIEXPORT jlong JNICALL Java_com_bytedance_hmp_Api_frame_1plane
  (JNIEnv *env, jclass, jlong ptr, jint p)
{
    JNI_PROTECT(
        return jlong(&jni::ptr<Frame>(ptr)->plane(p));
    )
    return 0;
}

/*
 * Class:     com_bytedance_hmp_Api
 * Method:    frame_to_device
 * Signature: (JLjava/lang/String;Z)J
 */
JNIEXPORT jlong JNICALL Java_com_bytedance_hmp_Api_frame_1to_1device
  (JNIEnv *env, jclass, jlong ptr, jstring device, jboolean non_blocking)
{
    JNI_PROTECT(
        auto dstr = jni::fromJString(env, device);
        return jni::makePtr<Frame>(jni::ptr<Frame>(ptr)->to(Device(dstr), non_blocking));
    )
    return 0;
}

/*
 * Class:     com_bytedance_hmp_Api
 * Method:    frame_copy_from
 * Signature: (JJ)V
 */
JNIEXPORT void JNICALL Java_com_bytedance_hmp_Api_frame_1copy_1from
  (JNIEnv *env, jclass, jlong ptr, jlong from)
{
    JNI_PROTECT(
        jni::ptr<Frame>(ptr)->copy_(*jni::ptr<Frame>(from));
    )
}

/*
 * Class:     com_bytedance_hmp_Api
 * Method:    frame_clone
 * Signature: (J)J
 */
JNIEXPORT jlong JNICALL Java_com_bytedance_hmp_Api_frame_1clone
  (JNIEnv *env, jclass, jlong ptr)
{
    JNI_PROTECT(
        return jni::makePtr<Frame>(jni::ptr<Frame>(ptr)->clone());
    )
    return 0;
}

/*
 * Class:     com_bytedance_hmp_Api
 * Method:    frame_crop
 * Signature: (JIIII)J
 */
JNIEXPORT jlong JNICALL Java_com_bytedance_hmp_Api_frame_1crop
  (JNIEnv *env, jclass, jlong ptr, jint left, jint top, jint width, jint height)
{
    JNI_PROTECT(
        return jni::makePtr<Frame>(jni::ptr<Frame>(ptr)->crop(left, top, width, height));
    )
    return 0;
}

/*
 * Class:     com_bytedance_hmp_Api
 * Method:    frame_to_image
 * Signature: (JI)J
 */
JNIEXPORT jlong JNICALL Java_com_bytedance_hmp_Api_frame_1to_1image
  (JNIEnv *env, jclass, jlong ptr, jint cformat)
{
    JNI_PROTECT(
        return jni::makePtr<Image>(jni::ptr<Frame>(ptr)->to_image((ChannelFormat)cformat));
    )
    return 0;
}

/*
 * Class:     com_bytedance_hmp_Api
 * Method:    frame_from_image
 * Signature: (JJ)J
 */
JNIEXPORT jlong JNICALL Java_com_bytedance_hmp_Api_frame_1from_1image
  (JNIEnv *env, jclass, jlong image, jlong pix_info)
{
    JNI_PROTECT(
        return jni::makePtr<Frame>(
            Frame::from_image(*jni::ptr<Image>(image),
            *jni::ptr<PixelInfo>(pix_info)));
    )
    return 0;
}

/*
 * Class:     com_bytedance_hmp_Api
 * Method:    frame_stringfy
 * Signature: (J)Ljava/lang/String;
 */
JNIEXPORT jstring JNICALL Java_com_bytedance_hmp_Api_frame_1stringfy
  (JNIEnv *env, jclass, jlong ptr)
{
    JNI_PROTECT(
        auto str = stringfy(*jni::ptr<Frame>(ptr));
        return jni::toJString(env, str);
    )
    return jni::toJString(env, "");
}


/*
 * Class:     com_bytedance_hmp_Api
 * Method:    image_make
 * Signature: (IIIIILjava/lang/String;Z)J
 */
JNIEXPORT jlong JNICALL Java_com_bytedance_hmp_Api_image_1make__IIIIILjava_lang_String_2Z
  (JNIEnv *env, jclass, jint width, jint height, jint channels, jint cformat,
   jint dtype, jstring device, jboolean pinned_memory)
{
    JNI_PROTECT(
        auto options = TensorOptions(jni::fromJString(env, device))
                                    .dtype((ScalarType)dtype)
                                    .pinned_memory(pinned_memory);

        return jni::makePtr<Image>(width, height, channels, 
                                   (ChannelFormat)cformat,
                                   options);
    )
    return 0;
}

/*
 * Class:     com_bytedance_hmp_Api
 * Method:    image_make
 * Signature: (JI)J
 */
JNIEXPORT jlong JNICALL Java_com_bytedance_hmp_Api_image_1make__JI
  (JNIEnv *env, jclass, jlong data, jint cformat)
{
    JNI_PROTECT(
        return jni::makePtr<Image>(*jni::ptr<Tensor>(data), (ChannelFormat)cformat);
    )
    return 0;
}

/*
 * Class:     com_bytedance_hmp_Api
 * Method:    image_make
 * Signature: (JIJ)J
 */
JNIEXPORT jlong JNICALL Java_com_bytedance_hmp_Api_image_1make__JIJ
  (JNIEnv *env, jclass, jlong data, jint cformat, jlong cm)
{
    JNI_PROTECT(
        return jni::makePtr<Image>(*jni::ptr<Tensor>(data), 
                                    (ChannelFormat)cformat,
                                    *jni::ptr<ColorModel>(cm));
    )
    return 0;
}

/*
 * Class:     com_bytedance_hmp_Api
 * Method:    image_free
 * Signature: (J)V
 */
JNIEXPORT void JNICALL Java_com_bytedance_hmp_Api_image_1free
  (JNIEnv *env, jclass, jlong ptr)
{
    JNI_PROTECT(
        jni::freePtr<Image>(ptr);
    )
}

/*
 * Class:     com_bytedance_hmp_Api
 * Method:    image_defined
 * Signature: (J)Z
 */
JNIEXPORT jboolean JNICALL Java_com_bytedance_hmp_Api_image_1defined
  (JNIEnv *env, jclass, jlong ptr)
{
    JNI_PROTECT(
        return jni::ptr<Image>(ptr)->operator bool();
    )
    return false;
}

/*
 * Class:     com_bytedance_hmp_Api
 * Method:    image_format
 * Signature: (J)I
 */
JNIEXPORT jint JNICALL Java_com_bytedance_hmp_Api_image_1format
  (JNIEnv *env, jclass, jlong ptr)
{
    JNI_PROTECT(
        return (jint)jni::ptr<Image>(ptr)->format();
    )
    return false;
}

/*
 * Class:     com_bytedance_hmp_Api
 * Method:    image_set_color_model
 * Signature: (JJ)V
 */
JNIEXPORT void JNICALL Java_com_bytedance_hmp_Api_image_1set_1color_1model
  (JNIEnv *env, jclass, jlong ptr, jlong cm)
{
    JNI_PROTECT(
        jni::ptr<Image>(ptr)->set_color_model(*jni::ptr<ColorModel>(cm));
    )
}

/*
 * Class:     com_bytedance_hmp_Api
 * Method:    image_color_model
 * Signature: (J)J
 */
JNIEXPORT jlong JNICALL Java_com_bytedance_hmp_Api_image_1color_1model
  (JNIEnv *env, jclass, jlong ptr)
{
    JNI_PROTECT(
        return (jlong)(&jni::ptr<Image>(ptr)->color_model());
    )
    return 0;
}

/*
 * Class:     com_bytedance_hmp_Api
 * Method:    image_wdim
 * Signature: (J)I
 */
JNIEXPORT jint JNICALL Java_com_bytedance_hmp_Api_image_1wdim
  (JNIEnv *env, jclass, jlong ptr)
{
    JNI_PROTECT(
        return jni::ptr<Image>(ptr)->wdim();
    )
    return 0;
}

/*
 * Class:     com_bytedance_hmp_Api
 * Method:    image_hdim
 * Signature: (J)I
 */
JNIEXPORT jint JNICALL Java_com_bytedance_hmp_Api_image_1hdim
  (JNIEnv *env, jclass, jlong ptr)
{
    JNI_PROTECT(
        return jni::ptr<Image>(ptr)->hdim();
    )
    return 0;
}

/*
 * Class:     com_bytedance_hmp_Api
 * Method:    image_cdim
 * Signature: (J)I
 */
JNIEXPORT jint JNICALL Java_com_bytedance_hmp_Api_image_1cdim
  (JNIEnv *env, jclass, jlong ptr)
{
    JNI_PROTECT(
        return jni::ptr<Image>(ptr)->cdim();
    )
    return 0;
}

/*
 * Class:     com_bytedance_hmp_Api
 * Method:    image_width
 * Signature: (J)I
 */
JNIEXPORT jint JNICALL Java_com_bytedance_hmp_Api_image_1width
  (JNIEnv *env, jclass, jlong ptr)
{
    JNI_PROTECT(
        return jni::ptr<Image>(ptr)->width();
    )
    return 0;
}

/*
 * Class:     com_bytedance_hmp_Api
 * Method:    image_height
 * Signature: (J)I
 */
JNIEXPORT jint JNICALL Java_com_bytedance_hmp_Api_image_1height
  (JNIEnv *env, jclass, jlong ptr)
{
    JNI_PROTECT(
        return jni::ptr<Image>(ptr)->height();
    )
    return 0;
}

/*
 * Class:     com_bytedance_hmp_Api
 * Method:    image_nchannels
 * Signature: (J)I
 */
JNIEXPORT jint JNICALL Java_com_bytedance_hmp_Api_image_1nchannels
  (JNIEnv *env, jclass, jlong ptr)
{
    JNI_PROTECT(
        return jni::ptr<Image>(ptr)->nchannels();
    )
    return 0;
}

/*
 * Class:     com_bytedance_hmp_Api
 * Method:    image_dtype
 * Signature: (J)I
 */
JNIEXPORT jint JNICALL Java_com_bytedance_hmp_Api_image_1dtype
  (JNIEnv *env, jclass, jlong ptr)
{
    JNI_PROTECT(
        return (jint)jni::ptr<Image>(ptr)->dtype();
    )
    return 0;
}

/*
 * Class:     com_bytedance_hmp_Api
 * Method:    image_device_type
 * Signature: (J)I
 */
JNIEXPORT jint JNICALL Java_com_bytedance_hmp_Api_image_1device_1type
  (JNIEnv *env, jclass, jlong ptr)
{
    JNI_PROTECT(
        return (jint)jni::ptr<Image>(ptr)->device().type();
    )
    return 0;
}

/*
 * Class:     com_bytedance_hmp_Api
 * Method:    image_device_index
 * Signature: (J)I
 */
JNIEXPORT jint JNICALL Java_com_bytedance_hmp_Api_image_1device_1index
  (JNIEnv *env, jclass, jlong ptr)
{
    JNI_PROTECT(
        return (jint)jni::ptr<Image>(ptr)->device().index();
    )
    return 0;
}

/*
 * Class:     com_bytedance_hmp_Api
 * Method:    image_data
 * Signature: (J)J
 */
JNIEXPORT jlong JNICALL Java_com_bytedance_hmp_Api_image_1data
  (JNIEnv *env, jclass, jlong ptr)
{
    JNI_PROTECT(
        return jlong(&jni::ptr<Image>(ptr)->data());
    )
    return 0;
}

/*
 * Class:     com_bytedance_hmp_Api
 * Method:    image_to_device
 * Signature: (JLjava/lang/String;Z)J
 */
JNIEXPORT jlong JNICALL Java_com_bytedance_hmp_Api_image_1to_1device
  (JNIEnv *env, jclass, jlong ptr, jstring device, jboolean non_blocking)
{
    JNI_PROTECT(
        auto dstr = jni::fromJString(env, device);
        return jni::makePtr<Image>(jni::ptr<Image>(ptr)->to(dstr, (bool)non_blocking));
    )
    return 0;
}

/*
 * Class:     com_bytedance_hmp_Api
 * Method:    image_to_dtype
 * Signature: (JI)J
 */
JNIEXPORT jlong JNICALL Java_com_bytedance_hmp_Api_image_1to_1dtype
  (JNIEnv *env, jclass, jlong ptr, jint dtype)
{
    JNI_PROTECT(
        return jni::makePtr<Image>(jni::ptr<Image>(ptr)->to((ScalarType)dtype));
    )
    return 0;
}

/*
 * Class:     com_bytedance_hmp_Api
 * Method:    image_copy_from
 * Signature: (JJ)V
 */
JNIEXPORT void JNICALL Java_com_bytedance_hmp_Api_image_1copy_1from
  (JNIEnv *env, jclass, jlong ptr, jlong from)
{
    JNI_PROTECT(
        jni::ptr<Image>(ptr)->copy_(*jni::ptr<Image>(from));
    )
}

/*
 * Class:     com_bytedance_hmp_Api
 * Method:    image_clone
 * Signature: (J)J
 */
JNIEXPORT jlong JNICALL Java_com_bytedance_hmp_Api_image_1clone
  (JNIEnv *env, jclass, jlong ptr)
{
    JNI_PROTECT(
        return jni::makePtr<Image>(jni::ptr<Image>(ptr)->clone());
    )
    return 0;
}

/*
 * Class:     com_bytedance_hmp_Api
 * Method:    image_crop
 * Signature: (JIIII)J
 */
JNIEXPORT jlong JNICALL Java_com_bytedance_hmp_Api_image_1crop
  (JNIEnv *env, jclass, jlong ptr, jint left, jint top, jint width, jint height)
{
    JNI_PROTECT(
        return jni::makePtr<Image>(jni::ptr<Image>(ptr)->crop(left, top, width, height));
    )
    return 0;
}

/*
 * Class:     com_bytedance_hmp_Api
 * Method:    image_stringfy
 * Signature: (J)Ljava/lang/String;
 */
JNIEXPORT jstring JNICALL Java_com_bytedance_hmp_Api_image_1stringfy
  (JNIEnv *env, jclass, jlong ptr)
{
    JNI_PROTECT(
        auto str = stringfy(*jni::ptr<Image>(ptr));
        return jni::toJString(env, str);
    )
    return jni::toJString(env, "");
}