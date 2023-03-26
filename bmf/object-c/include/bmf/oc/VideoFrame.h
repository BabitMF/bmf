#import <Foundation/Foundation.h>
#import <bmf/oc/Rational.h>
#import "bmf/oc/OpaqueDataKey.h"
#import "hmp/oc/Formats.h"
#import <hmp/oc/Device.h>
#import <hmp/oc/Scalar.h>
#import <hmp/oc/Image.h>
#import <hmp/oc/ScalarType.h>
#import <hmp/oc/Stream.h>
#import <hmp/oc/Tensor.h>

@interface BmfVideoFrame : NSObject

- (id)initFromPtr:(void*)vf own:(bool)own;
- (void) dealloc;

- (BmfVideoFrame*)init: (int)width height:(int)height channels:(int)channels format:(HmpChannelFormat)format dtype:(HmpScalarType)dtype device:(char *)device pinned_memory:(bool)pinned_memory;
- (BmfVideoFrame*)init: (int)width height:(int)height pix_info:(HmpPixelInfo*)pix_info device:(char *)device;
- (BmfVideoFrame*)initFromImage: (HmpImage*)image;
- (BmfVideoFrame*)initFromFrame: (HmpFrame*)frame;

- (bool)defined;
- (void*)ptr;
- (int)width;
- (int)height;
- (HmpScalarType)dtype;
- (bool)isImage;

- (HmpImage*)image;
- (HmpFrame*)frame;
- (BmfVideoFrame*)toImage:(HmpChannelFormat)channelFormat contiguous:(bool)contiguous;
- (BmfVideoFrame*)toFrame:(HmpPixelInfo*)pixInfo;

- (HmpDeviceType)deviceType;
- (int)deviceIndex;
- (BmfVideoFrame*)toDevice: (char *)device non_blocking:(bool)non_blocking;
- (BmfVideoFrame*)toDtype: (HmpScalarType)dtype;

- (void)copyFrom: (BmfVideoFrame*)from;
- (void)copyProps: (BmfVideoFrame*)from;
- (void)PrivateMerge: (BmfVideoFrame*)vf;
- (id)PrivateGet: (BmfOpaqueDataKey)key;
- (void)PrivateAttach: (BmfOpaqueDataKey)key option:(id)option;
- (void)setPts: (long)pts;
- (long)pts;
- (void)setTimeBase: (BmfRational*)rational;
- (BmfRational*)timeBase;
- (bool)Ready;
- (void)Record: (bool)use_current;
- (void)Synchronize;

@end
