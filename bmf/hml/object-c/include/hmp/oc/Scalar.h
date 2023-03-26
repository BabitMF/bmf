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
#ifndef _HMP_OC_SCALAR_H
#define _HMP_OC_SCALAR_H

#import <Foundation/Foundation.h>
#import <hmp/oc/ScalarType.h>


@interface HmpScalar: NSObject{
@protected
    void *_impl;
    bool _own;
}

- (instancetype) init;
- (instancetype) initFromBool: (bool) v;
- (instancetype) initFromInt: (int64_t) v;
- (instancetype) initFromFloat: (double) v;
- (instancetype) initFromPtr: (void*) ptr : (bool) own;

- (void*)ptr;

- (void) dealloc;

- (bool) to_bool;
- (int64_t) to_int;
- (double) to_float;

- (bool) is_integral: (bool) include_bool;
- (bool) is_floating_point;
- (bool) is_boolean;

@end

#endif