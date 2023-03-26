
#import <hmp/core/scalar.h>
#import <hmp/oc/Scalar.h>

using namespace hmp;

@interface HmpScalar()
@property (nonatomic, assign) Scalar* impl;
@property (nonatomic, assign) bool own;
@end


@implementation HmpScalar

- (instancetype) init
{
	self = [super init];
    if(self){
        self.impl = new Scalar();
        self.own = true;
    }
    return self;
}

- (instancetype) initFromBool: (bool) v
{
	self = [super init];
    if(self){
        self.impl = new Scalar(v);
        self.own = true;
    }
    return self;
}
- (instancetype) initFromInt: (int64_t) v
{

	self = [super init];
    if(self){
        self.impl = new Scalar(v);
        self.own = true;
    }
    return self;
}

- (instancetype) initFromFloat: (double) v
{
	self = [super init];
    if(self){
	self.impl = new Scalar(v);
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

- (bool) to_bool
{
    return self.impl->to<bool>();
}
- (int64_t) to_int
{
    return self.impl->to<int64_t>();
}

- (double) to_float
{
    return self.impl->to<double>();
}

- (bool) is_integral: (bool) include_bool
{
    return self.impl->is_integral(include_bool);
}

- (bool) is_floating_point
{
    return self.impl->is_floating_point();
}

- (bool) is_boolean
{
    return self.impl->is_boolean();
}

@end

