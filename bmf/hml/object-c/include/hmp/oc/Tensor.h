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
#ifndef _HMP_OC_TENSOR_H
#define _HMP_OC_TENSOR_H

#import <Foundation/Foundation.h>
#import <hmp/oc/Scalar.h>
#import <hmp/oc/ScalarType.h>
#import <hmp/oc/Device.h>

@interface HmpTensor: NSObject{
@protected
    void *_impl;
    bool _own;
}

+ (instancetype) empty: (NSMutableArray*) shape DType:(HmpScalarType) dtype Device:(NSString *)device Pinned: (bool) pinned_memory;
+ (instancetype) fromfile: (NSString*) fn : (HmpScalarType) dtype : (int64_t) count : (int64_t) offset;
+ (instancetype) from_buffer: (void*) data : (NSMutableArray*) shape : (HmpScalarType) dtype : (NSString*) device : (NSMutableArray*) strides;

- (instancetype) initFromPtr: (void*) ptr : (bool) own;

- (void*)ptr;

- (void) dealloc;

- (NSString*) description;

- (HmpTensor*) clone;

- (HmpTensor*) alias;

- (HmpTensor*) view: (NSMutableArray*) shape;

- (HmpTensor*) as_strided: (NSMutableArray*) shape : (NSMutableArray*) strides : (int64_t) offset;

- (HmpTensor*) permute: (NSMutableArray*) dims;

- (HmpTensor*) slice: (int64_t) dim : (int64_t) start : (int64_t) end : (int64_t) step;

- (HmpTensor*) select: (int64_t) dim : (int64_t) index;

- (HmpTensor*) reshape : (NSMutableArray*) shape;

- (bool) defined;

- (HmpDeviceType) device_type;

- (int64_t) device_index;

- (HmpScalarType) dtype;

- (NSMutableArray*) shape;

- (NSMutableArray*) strides;

- (int64_t) dim;

- (int64_t) size: (int64_t) dim;

- (int64_t) stride: (int64_t) dim;

- (int64_t) nbytes;

- (int64_t) itemsize;

- (int64_t) nitems;

- (bool) is_contiguous;

- (void*) unsafe_data;

- (HmpTensor*) fill_ : (HmpScalar*) value;

- (HmpTensor*) copy_ : (HmpTensor*) src;

- (HmpTensor*) contiguous;

- (void) tofile : (NSString*) fn;

@end

#endif