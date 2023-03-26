#import "hmp/tensor.h"
#import "hmp/oc/Tensor.h"

using namespace hmp;

@interface HmpTensor()
@property (nonatomic, assign) Tensor* impl;
@property (nonatomic, assign) bool own;
@end


@implementation HmpTensor

+ (instancetype) empty: (NSMutableArray*) shape DType:(HmpScalarType) dtype Device:(NSString *)device Pinned: (bool) pinned_memory
{
    std::vector<int64_t> vshape;
    for(NSNumber *num in shape){
        vshape.push_back([num longValue]);
    }

    TensorOptions options = TensorOptions((ScalarType)dtype)
                                .device(Device([device UTF8String]))
                                .pinned_memory(pinned_memory);
    Tensor* impl = new Tensor(hmp::empty(vshape, options));
    HmpTensor *tensor = [[HmpTensor alloc] initFromPtr: impl : true];
    if(!tensor){
        delete impl;
    }

    return tensor;
}


+ (instancetype) fromfile: (NSString*) fn : (HmpScalarType) dtype : (int64_t) count : (int64_t) offset
{
    Tensor* impl = new Tensor(hmp::fromfile([fn UTF8String], (ScalarType)dtype, count, offset));
    HmpTensor *tensor = [[HmpTensor alloc] initFromPtr : impl : true];
    if(!tensor){
        delete impl;
    }
    return tensor;
}


+ (instancetype) from_buffer: (void*) data : (NSMutableArray*) shape : (HmpScalarType) dtype : (NSString*) device : (NSMutableArray*) strides
{
    std::vector<int64_t> vshape;
    for(NSNumber *num in shape){
        vshape.push_back([num longValue]);
    }

    optional<SizeArray> vstrides;
    if(strides){
        SizeArray ss;
        for(NSNumber *num in strides){
            ss.push_back([num longValue]);
        }
        vstrides = ss;
    }

    DataPtr ptr = DataPtr(data, [](void*){}, Device([device UTF8String]));
    Tensor* impl = new Tensor(hmp::from_buffer(std::move(ptr), (ScalarType)dtype, vshape,  vstrides));
    HmpTensor *tensor = [[HmpTensor alloc] initFromPtr: impl : true];
    if(!tensor){
        delete impl;
    }

    return tensor;
}

- (instancetype) initFromPtr: (void*) ptr : (bool) own
{
    self = [super init];
    if(self){
        self.impl = (Tensor*)ptr;
        self.own = own;
    }
    return self;
}

- (void*)ptr
{
    return self.impl;
}

- (void) dealloc
{
    if(self.own && self.impl){
        delete self.impl;
    }
}


- (NSString*) description
{
    return [NSString stringWithFormat: @"%s" , self.impl->repr().c_str()];
}


- (HmpTensor*) clone
{
    Tensor *impl = new Tensor(self.impl->clone());
    HmpTensor *ret = [[HmpTensor alloc] initFromPtr : impl : true];
    if(!ret){
        delete impl;
    }
    return ret;
}

- (HmpTensor*) alias
{
    Tensor *impl = new Tensor(self.impl->alias());
    HmpTensor *ret = [[HmpTensor alloc] initFromPtr : impl : true];
    if(!ret){
        delete impl;
    }
    return ret;
}


- (HmpTensor*) view: (NSMutableArray*) shape
{
    std::vector<int64_t> vshape;
    for(NSNumber *num in shape){
        vshape.push_back([num longValue]);
    }
    Tensor *impl = new Tensor(self.impl->view(vshape));
    HmpTensor *ret = [[HmpTensor alloc] initFromPtr : impl : true];
    if(!ret){
        delete impl;
    }
    return ret;
}

- (HmpTensor*) as_strided: (NSMutableArray*) shape : (NSMutableArray*) strides : (int64_t) offset
{
    std::vector<int64_t> vshape;
    for(NSNumber *num in shape){
        vshape.push_back([num longValue]);
    }

    std::vector<int64_t> vstrides;
    for(NSNumber *num in strides){
        vstrides.push_back([num longValue]);
    }

    Tensor *impl = new Tensor(self.impl->as_strided(vshape, vstrides, offset));
    HmpTensor *ret = [[HmpTensor alloc] initFromPtr : impl : true];
    if(!ret){
        delete impl;
    }
    return ret;
}

- (HmpTensor*) permute: (NSMutableArray*) dims
{
    std::vector<int64_t> vdims;
    for(NSNumber *num in dims){
        vdims.push_back([num longValue]);
    }
    Tensor *impl = new Tensor(self.impl->permute(vdims));
    HmpTensor *ret = [[HmpTensor alloc] initFromPtr : impl : true];
    if(!ret){
        delete impl;
    }
    return ret;
}


- (HmpTensor*) slice: (int64_t) dim : (int64_t) start : (int64_t) end : (int64_t) step
{
    Tensor *impl = new Tensor(self.impl->slice(dim, start, end, step));
    HmpTensor *ret = [[HmpTensor alloc] initFromPtr : impl : true];
    if(!ret){
        delete impl;
    }
    return ret;
}

- (HmpTensor*) select: (int64_t) dim : (int64_t) index
{
    Tensor *impl = new Tensor(self.impl->select(dim, index));
    HmpTensor *ret = [[HmpTensor alloc] initFromPtr : impl : true];
    if(!ret){
        delete impl;
    }
    return ret;
}

- (HmpTensor*) reshape : (NSMutableArray*) shape
{
    std::vector<int64_t> vshape;
    for(NSNumber *num in shape){
        vshape.push_back([num longValue]);
    }

    Tensor *impl = new Tensor(self.impl->reshape(vshape));
    HmpTensor *ret = [[HmpTensor alloc] initFromPtr : impl : true];
    if(!ret){
        delete impl;
    }
    return ret;
}

- (bool) defined
{
    return self.impl->defined();
}

- (HmpDeviceType) device_type
{
    return (HmpDeviceType)self.impl->device_type();
}

- (int64_t) device_index
{
    return self.impl->device_index();
}

- (HmpScalarType) dtype
{
    return (HmpScalarType)self.impl->dtype();
}


- (NSMutableArray*) shape
{
    std::vector<int64_t> vshape = self.impl->shape();
    NSMutableArray *array = [[NSMutableArray alloc] initWithCapacity : vshape.size()];
    for(auto &v : vshape){
        [array addObject : [NSNumber numberWithLong : v]];
    }
    return array;
}


- (NSMutableArray*) strides
{
    std::vector<int64_t> vstrides = self.impl->strides();
    NSMutableArray *array = [[NSMutableArray alloc] initWithCapacity : vstrides.size()];
    for(auto &v : vstrides){
        [array addObject : [NSNumber numberWithLong : v]];
    }
    return array;
}


- (int64_t) dim
{
    return self.impl->dim();
}

- (int64_t) size: (int64_t) dim
{
    return self.impl->size(dim);
}

- (int64_t) stride: (int64_t) dim
{
    return self.impl->stride(dim);
}

- (int64_t) nbytes
{
    return self.impl->nbytes();
}

- (int64_t) itemsize
{
    return self.impl->itemsize();
}

- (int64_t) nitems
{
    return self.impl->nitems();
}

- (bool) is_contiguous
{
    return self.impl->is_contiguous();
}

- (void*) unsafe_data
{
    return self.impl->unsafe_data();
}

- (HmpTensor*) fill_ : (HmpScalar*) value
{
    self.impl->fill_(*(Scalar*)[value ptr]);
    return self;
}

- (HmpTensor*) copy_ : (HmpTensor*) src
{
    self.impl->copy_(*(Tensor*)[src ptr]);
    return self;
}

- (HmpTensor*) contiguous
{
    Tensor *impl = new Tensor(self.impl->contiguous());
    HmpTensor *ret = [[HmpTensor alloc] initFromPtr : impl : true];
    if(!ret){
        delete impl;
    }
    return ret;
}


- (void) tofile : (NSString*) fn
{
    self.impl->tofile([fn UTF8String]);
}

@end
