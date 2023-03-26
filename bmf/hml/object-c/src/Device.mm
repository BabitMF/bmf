
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
