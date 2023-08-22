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

#import "hmp/core/stream.h"
#import "hmp/oc/Stream.h"

using namespace hmp;

@interface HmpStream()
@property (nonatomic, assign) Stream* impl;
@property (nonatomic, assign) bool own;
@end


@implementation HmpStream


+ (instancetype) create: (HmpDeviceType) device_type Flags:(uint64_t)flags
{
	HmpStream *stream = [HmpStream alloc];
    if(stream){
        stream.impl = new Stream(create_stream((DeviceType)device_type, flags));
        stream.own = true;
    }
    return stream;
}


+ (instancetype) current: (HmpDeviceType) device_type
{
	HmpStream *stream = [HmpStream alloc];
    if(stream){
        stream.impl = new Stream(current_stream((DeviceType)device_type).value());
        stream.own = true;
    }
    return stream;
}

+ (void) set_current: (HmpStream*) stream
{
    set_current_stream(*stream.impl);
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


- (bool) isEqual: (HmpStream*) stream
{
    return (*stream.impl) == (*self.impl);
}

- (bool) query
{
    return self.impl->query();
}

- (void) synchronize
{
    self.impl->synchronize();
}

- (HmpDevice*) device
{
    Device* dptr = new Device(self.impl->device());
    HmpDevice *dev = [[HmpDevice alloc] initFromPtr: dptr : true];
    if(!dev){
        delete dptr;
    }
    return dev;
}

@end
