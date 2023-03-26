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
#ifndef _HMP_OC_STREAM_H
#define _HMP_OC_STREAM_H

#import <Foundation/Foundation.h>
#import <hmp/oc/Device.h>


@interface HmpStream: NSObject{
@protected
    void *_impl;
    bool _own;
}

+ (instancetype) create: (HmpDeviceType) device_type Flags: (uint64_t) flags;
+ (instancetype) current: (HmpDeviceType) device_type;
+ (void) set_current: (HmpStream*) stream;

- (instancetype) initFromPtr: (void*) ptr : (bool) own;

- (void*)ptr;

- (void) dealloc;

- (bool) isEqual: (HmpStream*) stream;

- (bool) query;

- (void) synchronize;

- (HmpDevice*) device;


@end

#endif