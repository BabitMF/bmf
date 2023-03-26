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
#include <hmp/core/ref_ptr.h>

namespace hmp{

RefObject::~RefObject()
{
    auto refcount = refcount_.load();

    //sanit check
    if(refcount){
        HMP_ERR("RefObject: invalid state of RefObject {}, refcount={}", (void*)this, refcount);
    }
}



} //namespace