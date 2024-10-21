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

#import "hmp/core/device.h"
#import "hmp/oc/Device.h"

using namespace hmp;


@interface HmpDevice()
@property (nonatomic, assign) Device* impl;
@property (nonatomic, assign) bool own;
@end


@implementation HmpDevice


+ (int64_t) count: (HmpDeviceType) type
{
    return device_count((DeviceType)type);
}

+ (int) current: (HmpDeviceType) type
{
    return current_device((DeviceType)type).value().index();
}

+ (void) set_current: (HmpDevice*) device
{
    return set_current_device(*(Device*)device.impl);
}


- (instancetype) init
{
	self = [super init];
    if(self){
        self.impl = new Device();
        self.own = true;
    }
    return self;
}


- (instancetype) initFromString: (NSString*) device
{
    self = [super init];
    if(self){
        self.impl = new Device([device UTF8String]);
        self.own = true;
    }
    return self;
}

- (instancetype) initFromPtr: (void*) ptr : (bool) own
{
    self = [super init];
    if(self){
        self->_impl = ptr;
        self->_own = own;
    }
    return self;
}

- (void*)ptr
{
	return self->_impl;
}

- (void) dealloc
{
    if(self.own && self.impl){
	    delete self.impl;
    }
}

- (NSString *)description
{
    return [NSString stringWithFormat: @"%s" , stringfy(*self.impl).c_str()];
}

- (HmpDeviceType) type
{
    return (HmpDeviceType)self.impl->type();
}

- (int) index
{
    return self.impl->index();
}


@end
