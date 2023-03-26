#import "bmf/oc/VideoFrame.h"
#import "bmf/sdk/module_functor.h"
#import "bmf/sdk/video_frame.h"
#import "bmf/sdk/json_param.h"

using namespace bmf_sdk;
using namespace hmp;

@interface BmfVideoFrame()
@property (nonatomic, assign) VideoFrame* p;
@property (nonatomic, assign) bool own;
@end

@implementation BmfVideoFrame

- (id)initFromPtr:(void*)vf own:(bool)own{
    if((self = [super init]) != nil){
        self.p = (VideoFrame*)vf;
        self.own = own;
    }
    return self;
}

- (void) dealloc
{
    if(self.own && self.p){
        delete self.p;
    }
}

- (void*)ptr{
    return (void*)self.p;
}

- (BmfVideoFrame*)initFromImage: (HmpImage*)image
{
    if((self = [super init]) != nil){
        Image *img = (Image*)[image ptr];
        VideoFrame *vf = new VideoFrame(*img);
        self.p = vf;
        self.own = true;
    }
    return self;
}

- (BmfVideoFrame*)initFromFrame: (HmpFrame*)frame
{
    if((self = [super init]) != nil){
        Frame *frm = (Frame*)[frame ptr];
        VideoFrame *vf = new VideoFrame(*frm);
        self.p = vf;
        self.own = true;
    }
    return self;
}

- (BmfVideoFrame*)init: (int)width height:(int)height channels:(int)channels format:(HmpChannelFormat)format dtype:(HmpScalarType)dtype device:(char *)device pinned_memory:(bool)pinned_memory
{
    if((self = [super init]) != nil){
        VideoFrame *vf = new VideoFrame(width, height, channels, (ChannelFormat)(format), TensorOptions((ScalarType)(dtype))
                                        .device(Device(device)).pinned_memory(pinned_memory));
        self.p = vf;
        self.own = true;
    }
    return self;
}

- (BmfVideoFrame*)init: (int)width height:(int)height pix_info:(HmpPixelInfo*)pix_info device:(char *)device
{
    if((self = [super init]) != nil){
        PixelInfo *pi = (PixelInfo*)[pix_info ptr];
        VideoFrame *vf = new VideoFrame(width, height, *pi, Device(device));
        self.p = vf;
        self.own = true;
    }
    return self;
}

- (bool)defined{
    return (bool)self.p;
}

- (int)width{
    return self.p->width();
}

- (int)height{
    return self.p->height();
}

- (HmpScalarType)dtype{
    return (HmpScalarType)self.p->dtype();
}

- (bool)isImage{
    return self.p->is_image();
}

- (HmpImage*)image{
    Image *img = (Image*)&self.p->image();
    return [[HmpImage alloc] initFromPtr:img : false];
}

- (HmpFrame*)frame{
    Frame *frm = (Frame*)&self.p->frame();
    return [[HmpFrame alloc] initFromPtr:frm : false];
}

- (BmfVideoFrame*)toImage:(HmpChannelFormat)channelFormat contiguous:(bool)contiguous{
    auto tmp = self.p->to_image((ChannelFormat)channelFormat, contiguous);
    VideoFrame *vf = new VideoFrame(tmp);
    return [[BmfVideoFrame alloc] initFromPtr:vf own:true];
}

- (BmfVideoFrame*)toFrame:(HmpPixelInfo*)pixInfo{
    const PixelInfo *pix_info = (const PixelInfo*)[pixInfo ptr];
    int f = [pixInfo format];
    auto tmp = self.p->to_frame(*pix_info);
    VideoFrame *vf = new VideoFrame(tmp);
    return [[BmfVideoFrame alloc] initFromPtr:vf own:true];
}

- (HmpDeviceType)deviceType{
    return (HmpDeviceType)self.p->device().type();
}

- (int)deviceIndex{
    return self.p->device().index();
}

- (BmfVideoFrame*)toDevice: (char *)device non_blocking:(bool)non_blocking{
    VideoFrame *vf = new VideoFrame(self.p->to(Device(device), non_blocking));
    return [[BmfVideoFrame alloc] initFromPtr:vf own:true];
}

- (BmfVideoFrame*)toDtype: (HmpScalarType)dtype{
    VideoFrame *vf = new VideoFrame(self.p->to((ScalarType)dtype));
    return [[BmfVideoFrame alloc] initFromPtr:vf own:true];
}

- (void)copyFrom: (BmfVideoFrame*)from{
    VideoFrame *v = (VideoFrame*)[from ptr];
    self.p->copy_(*v);
}

- (void)copyProps: (BmfVideoFrame*)from{
    VideoFrame *v = (VideoFrame*)[from ptr];
    self.p->copy_props(*v);
}

- (void)PrivateMerge: (BmfVideoFrame*)vf{
    VideoFrame *from = (VideoFrame*)[vf ptr];
    self.p->private_merge(*from);
}

- (id)PrivateGet: (BmfOpaqueDataKey)key{
    if (key == kJsonParam){
        JsonParam *json = (JsonParam*)self.p->private_get<JsonParam>();
        auto str = json->dump();
        char* jsonParamStr = strdup(str.c_str());
        NSString *string = [[NSString alloc] initWithCString:jsonParamStr encoding:NSUTF8StringEncoding];
        NSError *error = nil;
        NSData *data= [string dataUsingEncoding:NSUTF8StringEncoding];
        id jsonObject = [NSJSONSerialization JSONObjectWithData:data options:NSJSONReadingAllowFragments error:&error];
        return jsonObject;
    }else{
        return nil;
    }
}

- (void)PrivateAttach: (BmfOpaqueDataKey)key option:(id)option{
    if (key == kJsonParam){
        if([NSJSONSerialization isValidJSONObject:option]){
            NSError *error = nil;
            NSData *data_str = [NSJSONSerialization dataWithJSONObject:option options:NSJSONReadingAllowFragments error:&error];
            NSString *ns_string = [[NSString alloc]initWithData:data_str encoding:NSUTF8StringEncoding];
            const char *str =[ns_string UTF8String];
            JsonParam *j = new JsonParam();
            j->parse(str);
            self.p->private_attach<JsonParam>(j);
        }else{
            NSLog(@"private attach did not get an actual jsonparam.");
        }
    }
}

- (void)setPts: (long)pts{
    self.p->set_pts(pts);
}

- (long)pts{
    return self.p->pts();
}

- (void)setTimeBase: (BmfRational*)rational{
    self.p->set_time_base(Rational([rational num], [rational den]));
}

- (BmfRational*)timeBase{
    int num = self.p->time_base().num;
    int den = self.p->time_base().den;
    return [[BmfRational alloc] init:num den:den];
}

- (bool)Ready{
    return self.p->ready();
}

- (void)Record: (bool)use_current{
    self.p->record(use_current);
}

- (void)Synchronize{
    self.p->synchronize();
}

@end
