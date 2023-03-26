#import "test_VideoFrame.h"
#import <hmp/oc/Device.h>
#import "hmp/oc/Formats.h"
#import <hmp/oc/Device.h>
#import <hmp/oc/Scalar.h>
#import <hmp/oc/Image.h>
#import <hmp/oc/ScalarType.h>
#import <hmp/oc/Stream.h>
#import <hmp/oc/Tensor.h>
#import "bmf/oc/OpaqueDataKey.h"
#import <bmf/oc/Rational.h>
#import "bmf/oc/VideoFrame.h"

@implementation BmfVideoFrameTests

- (int) testAll
{
    // default constructors
    BmfVideoFrame *vf0 = [[BmfVideoFrame alloc]init];
    if ([vf0 defined] != 0) {
        return 1;
    }
    
    // from image
    NSMutableArray *shape = [NSMutableArray arrayWithCapacity: 3];
    [shape addObject : [NSNumber numberWithLong : 2]];
    [shape addObject : [NSNumber numberWithLong : 3]];
    [shape addObject : [NSNumber numberWithLong : 4]];
    HmpTensor *tensor = [HmpTensor empty: shape DType:kInt32 Device:@"cpu" Pinned:false];
    HmpImage *image = [[HmpImage alloc] init:tensor :HmpChannelFormat::kNCHW];
    BmfVideoFrame *vf_from_img = [[BmfVideoFrame alloc]initFromImage:image];
    if ([vf_from_img width] != 4 || [vf_from_img height] != 3) {
        return 2;
    }
    
    // from frame
    HmpPixelInfo *pixInfo = [[HmpPixelInfo alloc]init:HmpPixelFormat::PF_RGB24];
    HmpFrame *frame = [[HmpFrame alloc]init:100 :200 :pixInfo];
    BmfVideoFrame *vf_from_frm = [[BmfVideoFrame alloc] initFromFrame:frame];
    if ([vf_from_frm width] != 100 || [vf_from_frm height] != 200) {
        return 3;
    }
    
    // as image
    BmfVideoFrame *vf_as_img = [[BmfVideoFrame alloc] init:1080 height:720 channels:3 format:HmpChannelFormat::kNHWC dtype:HmpScalarType::kUInt8 device:"cpu" pinned_memory:false];
    if ([vf_as_img defined] != 1 || [vf_as_img width] != 1080 || [vf_as_img height] !=720 || [vf_as_img dtype] != HmpScalarType::kUInt8 || [vf_as_img isImage] != 1 || [vf_as_img deviceType] != HmpDeviceType::kCPU || [vf_as_img deviceIndex] != 0) {
        return 4;
    }

    // get image
    HmpImage *hmp_image = [vf_as_img image];
    if ([hmp_image width] != 1080 || [hmp_image height] != 720) {
        return 5;
    }
    
    // to frame
    BmfVideoFrame *vf_to_frm = [vf_as_img toFrame:pixInfo];
    if ([vf_to_frm width] != 1080 || [vf_to_frm height] != 720) {
        return 6;
    }
    
    // as frame
    BmfVideoFrame *vf_as_frm = [[BmfVideoFrame alloc] init:1920 height:1440 pix_info:pixInfo device:"cpu"];
    if ([vf_as_frm width]!=1920 || [vf_as_frm height]!=1440) {
        return 7;
    }

    // get frame
    HmpFrame *hmp_frame = [vf_as_frm frame];
    if ([hmp_frame width]!=1920 || [hmp_frame height]!=1440) {
        return 8;
    }
    
    // to image
    BmfVideoFrame *vf_to_img = [vf_as_frm toImage:HmpChannelFormat::kNHWC contiguous:true];
    if ([vf_to_img width]!=1920 || [vf_to_img height]!=1440) {
        return 9;
    }
    
    // pts & timebase
    BmfVideoFrame *vf1 = [[BmfVideoFrame alloc] init:1080 height:720 channels:3 format:HmpChannelFormat::kNCHW dtype:HmpScalarType::kUInt8 device:"cpu" pinned_memory:false];
    [vf1 setPts:9876];
    BmfRational *rational = [[BmfRational alloc] init:1 den:100];
    [vf1 setTimeBase:rational];
    if ([vf1 pts] != 9876 || [[vf1 timeBase] num] != 1 || [[vf1 timeBase] den] != 100) {
        return 10;
    }
            
    // copyProps
    [vf_as_img copyProps:vf1];
    if ([vf_as_img pts] != 9876 || [[vf_as_img timeBase] num] != 1 || [[vf_as_img timeBase] den] != 100) {
        return 11;
    }
    
    // private attach & private get
    char* ch = "{\"name\":\"ios_sdk\",\"option\":{\"path\":\"my_path\",\"entry\":\"my_entry\"}}";
    NSString *string = [[NSString alloc] initWithCString:ch encoding:NSUTF8StringEncoding];
    NSError *error = nil;
    NSData *data= [string dataUsingEncoding:NSUTF8StringEncoding];
    id jsonObject = [NSJSONSerialization JSONObjectWithData:data options:NSJSONReadingAllowFragments error:&error];
    [vf_as_img PrivateAttach:kJsonParam option:jsonObject];
    id obj = [vf_as_img PrivateGet:kJsonParam];
    NSError *error2 = nil;
    NSData *data_str = [NSJSONSerialization dataWithJSONObject:obj options:NSJSONReadingAllowFragments error:&error2];
    NSString *ns_string = [[NSString alloc]initWithData:data_str encoding:NSUTF8StringEncoding];
    const char *str =[ns_string UTF8String];
    if ([jsonObject isEqual: obj] != 1){
        return 12;
    }
    
    // private merge
    [vf1 PrivateMerge:vf_as_img];
    id obj2 = [vf1 PrivateGet:kJsonParam];
    if ([jsonObject isEqual: obj2] != 1){
        return 13;
    }
    
    return 0;
}

@end
