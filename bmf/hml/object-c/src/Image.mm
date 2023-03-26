#import "hmp/imgproc/image.h"

#import "hmp/oc/Image.h"

using namespace hmp;


@interface HmpImage()
@property (nonatomic, assign) Image* impl;
@property (nonatomic, assign) bool own;
@end


@implementation HmpImage


- (instancetype) init : (HmpTensor*) data : (HmpChannelFormat) format
{
	self = [super init];
    if(self){
        self.impl = new Image(*(Tensor*)[data ptr], (ChannelFormat)format);
        self.own = true;
    }
    return self;
}

- (instancetype) init : (int) width : (int) height : (int) nchannels : (HmpChannelFormat) format : (HmpScalarType) dtype
{
    self = [super init];
    if(self){
        self.impl = new Image(width, height, nchannels, (ChannelFormat)format, (ScalarType)dtype);
        self.own = true;
    }
    return self;
}

- (instancetype) initFromPtr: (void*) ptr : (bool) own
{
    self = [super init];
    if(self){
        self.impl = (Image*)ptr;
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

- (bool) defined
{
    return self.impl->operator bool();
}

- (NSString*) description
{
    return [NSString stringWithFormat: @"%s" , stringfy(*self.impl).c_str()];
}


- (HmpChannelFormat) format
{
    return (HmpChannelFormat)self.impl->format();
}

- (void) set_color_model: (HmpColorModel*) cm
{
    self.impl->set_color_model(*(ColorModel*)[cm ptr]);
}

- (HmpColorModel*) color_model
{
    return [[HmpColorModel alloc] initFromPtr : (void*)&self.impl->color_model() : false];
}

- (int) wdim
{
    return self.impl->wdim();
}

- (int) hdim
{
    return self.impl->hdim();
}

- (int) cdim
{
    return self.impl->cdim();
}

- (int) width
{
    return self.impl->width();
}

- (int) height
{
    return self.impl->height();
}

- (int) nchannels
{
    return self.impl->nchannels();
}

- (HmpScalarType) dtype
{
    return (HmpScalarType)self.impl->dtype();
}


- (HmpDevice*) device
{
    return [[HmpDevice alloc] initFromPtr : (void*)&self.impl->device() : false];
}

- (HmpTensor*) data
{
    return [[HmpTensor alloc] initFromPtr : (void*)&self.impl->data() : false];
}

- (void*) unsafe_data
{
    return self.impl->unsafe_data();
}

- (HmpImage*) copy_ : (HmpImage*) from
{
    self.impl->copy_(*(Image*)[from ptr]);
    return self;
}

- (HmpImage*) clone
{
    Image* impl = new Image(self.impl->clone());
    HmpImage *ret = [[HmpImage alloc] initFromPtr : impl : true];
    if(!ret){
        delete impl;
    }
    return ret;
}

- (HmpImage*) crop : (int)left : (int)top : (int)width : (int) height
{
    Image* impl = new Image(self.impl->crop(left, top, width, height));
    HmpImage* ret = [[HmpImage alloc] initFromPtr : impl : true];
    if(!ret){
        delete impl;
    }
    return ret;
}

- (HmpImage*) select : (int) channel
{
    Image* impl = new Image(self.impl->select(channel));
    HmpImage* ret = [[HmpImage alloc] initFromPtr : impl : true];
    if(!ret){
        delete impl;
    }
    return ret;
}

@end //HmpImage


@interface HmpFrame()
@property (nonatomic, assign) Frame* impl;
@property (nonatomic, assign) bool own;
@property (nonatomic, assign) CVPixelBufferRef buffer;
@end


@implementation HmpFrame

- (instancetype) init : (int) width : (int) height : (HmpPixelInfo*) pix_info
{
    self = [super init];
    if(self){
        self.impl = new Frame(width, height, *(PixelInfo*)[pix_info ptr]);
        self.own = true;
        self.buffer = nil;
    }
    return self;
}

- (instancetype) initFromPtr: (void*) ptr : (bool) own
{
    self = [super init];
    if(self){
        self.impl = (Frame*)ptr;
        self.own = own;
        self.buffer = nil;
    }
    return self;
}


- (instancetype) initFromPixelBuffer: (CVPixelBufferRef) pix_buffer
{
    auto format = CVPixelBufferGetPixelFormatType(pix_buffer);

    //supported pixel format: https://developer.apple.com/library/archive/qa/qa1501/_index.html
    Frame *impl = nil;
    //NV12
    auto noop = [](void*){};
    if(format == kCVPixelFormatType_420YpCbCr8BiPlanarFullRange 
        || format == kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange){
        //
        auto ydata = DataPtr(CVPixelBufferGetBaseAddressOfPlane(pix_buffer, 0), noop, {});
        auto uvdata = DataPtr(CVPixelBufferGetBaseAddressOfPlane(pix_buffer, 1), noop, {});
        int64_t width = CVPixelBufferGetWidth(pix_buffer);
        int64_t height = CVPixelBufferGetHeight(pix_buffer);
        int64_t ystride = CVPixelBufferGetBytesPerRowOfPlane(pix_buffer, 0);
        int64_t uvstride = CVPixelBufferGetBytesPerRowOfPlane(pix_buffer, 1);

        auto Y = from_buffer(std::move(ydata), hmp::kUInt8, SizeArray{height, width, 1}, SizeArray{ystride, 1, 1});
        auto UV = from_buffer(std::move(uvdata), hmp::kUInt8, SizeArray{height/2, width/2, 2}, SizeArray{uvstride, 2, 1});
        auto crange = format == kCVPixelFormatType_420YpCbCr8BiPlanarFullRange ? hmp::CR_JPEG : hmp::CR_MPEG;
        PixelInfo pix_info = PixelInfo(hmp::PF_NV12, hmp::CS_UNSPECIFIED, crange);

        impl = new Frame(TensorList{Y, UV}, (int)width, (int)height, pix_info);
    }
    //YUV420P
    else if(format == kCVPixelFormatType_420YpCbCr8Planar){
        //
        auto ydata = DataPtr(CVPixelBufferGetBaseAddressOfPlane(pix_buffer, 0), noop, {});
        auto udata = DataPtr(CVPixelBufferGetBaseAddressOfPlane(pix_buffer, 1), noop, {});
        auto vdata = DataPtr(CVPixelBufferGetBaseAddressOfPlane(pix_buffer, 2), noop, {});
        int64_t width = CVPixelBufferGetWidth(pix_buffer);
        int64_t height = CVPixelBufferGetHeight(pix_buffer);
        int64_t ystride = CVPixelBufferGetBytesPerRowOfPlane(pix_buffer, 0);
        int64_t ustride = CVPixelBufferGetBytesPerRowOfPlane(pix_buffer, 1);
        int64_t vstride = CVPixelBufferGetBytesPerRowOfPlane(pix_buffer, 2);

        auto Y = from_buffer(std::move(ydata), hmp::kUInt8, SizeArray{height, width, 1}, SizeArray{ystride, 1, 1});
        auto U = from_buffer(std::move(udata), hmp::kUInt8, SizeArray{height/2, width/2, 1}, SizeArray{ustride, 1, 1});
        auto V = from_buffer(std::move(vdata), hmp::kUInt8, SizeArray{height/2, width/2, 1}, SizeArray{vstride, 1, 1});
        PixelInfo pix_info = PixelInfo(hmp::PF_YUV420P);

        impl = new Frame(TensorList{Y, U, V}, (int)width, (int)height, pix_info);
    }
    else if(format == kCVPixelFormatType_32BGRA){
        auto dataPtr = DataPtr(CVPixelBufferGetBaseAddressOfPlane(pix_buffer, 0), noop, {});
        int64_t width = CVPixelBufferGetWidth(pix_buffer);
        int64_t height = CVPixelBufferGetHeight(pix_buffer);
        int64_t stride = CVPixelBufferGetBytesPerRowOfPlane(pix_buffer, 0);

        auto data = from_buffer(std::move(dataPtr), hmp::kUInt8, SizeArray{height, width, 4}, SizeArray{stride, 4, 1});
        
        PixelInfo pix_info = PixelInfo(hmp::PF_BGRA32);

        impl = new Frame(TensorList{data}, (int)width, (int)height, pix_info);
    }
    else if(format == kCVPixelFormatType_24RGB){
        auto dataPtr = DataPtr(CVPixelBufferGetBaseAddressOfPlane(pix_buffer, 0), noop, {});
        int64_t width = CVPixelBufferGetWidth(pix_buffer);
        int64_t height = CVPixelBufferGetHeight(pix_buffer);
        int64_t stride = CVPixelBufferGetBytesPerRowOfPlane(pix_buffer, 0);

        auto data = from_buffer(std::move(dataPtr), hmp::kUInt8, SizeArray{height, width, 3}, SizeArray{stride, 3, 1});
        
        PixelInfo pix_info = PixelInfo(hmp::PF_RGB24);

        impl = new Frame(TensorList{data}, (int)width, (int)height, pix_info);
    }

    if(impl){
        if(self = [super init]){
            self.impl = impl;
            self.own = true;
            self.buffer = pix_buffer;
            CVPixelBufferRetain(pix_buffer);
        }
        else{
            delete impl;
            impl = nil;
        }
    }

    return self;
}

- (void*)ptr
{
    return self.impl;
}

- (void) dealloc
{
    if(self.buffer){
        CVPixelBufferRelease(self.buffer);
    }

    if(self.own && self.impl){
        delete self.impl;
    }
}

- (bool) defined
{
    return self.impl->operator bool();
}

- (NSString*) description
{
    return [NSString stringWithFormat: @"%s" , stringfy(*self.impl).c_str()];
}

- (HmpPixelInfo*) pix_info
{
    return [[HmpPixelInfo alloc] initFromPtr : (void*)&self.impl->pix_info() : false];
}


- (HmpPixelFormat) format
{
    return (HmpPixelFormat)self.impl->format();
}

- (int) width
{
    return self.impl->width();
}

- (int) height
{
    return self.impl->height();
}

- (HmpScalarType) dtype
{
    return (HmpScalarType)self.impl->dtype();
}

- (HmpDevice*) device
{
    return [[HmpDevice alloc] initFromPtr : (void*)&self.impl->device() : false];
}

- (int64_t) nplanes
{
    return self.impl->nplanes();
}

- (HmpTensor*) plane: (int) p
{
    return [[HmpTensor alloc] initFromPtr : (void*)&self.impl->plane(p) : false];
}

- (void*) plane_data : (int) p
{
    return self.impl->plane_data(p);
}
- (HmpFrame*) copy_: (HmpFrame*) from
{
    self.impl->copy_(*(Frame*)[from ptr]);
    return self;
}

- (HmpFrame*) crop : (int)left : (int)top : (int)width : (int) height
{
    Frame* impl = new Frame(self.impl->crop(left, top, width, height));
    HmpFrame* ret = [[HmpFrame alloc] initFromPtr : impl : true];
    if(!ret){
        delete impl;
    }
    return ret;
}

- (HmpImage*) to_image : (HmpChannelFormat) cformat
{
    Image* impl = new Image(self.impl->to_image((ChannelFormat)cformat));
    HmpImage* ret = [[HmpImage alloc] initFromPtr : impl : true];
    if(!ret){
        delete impl;
    }
    return ret;
}

+ (HmpFrame*) from_image : (HmpImage*) image : (HmpPixelInfo*) pix_info
{
    Frame *impl = new Frame(Frame::from_image(*(Image*)[image ptr], *(PixelInfo*)[pix_info ptr]));
    HmpFrame* ret = [[HmpFrame alloc] initFromPtr : impl : true];
    if(!ret){
        delete impl;
    }
    return ret;
}

@end //HmpFrame
