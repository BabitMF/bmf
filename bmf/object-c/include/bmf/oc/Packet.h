#import <Foundation/Foundation.h>
#import <bmf/oc/VideoFrame.h>
#import <bmf/oc/AudioFrame.h>

@interface BmfPacket: NSObject

- (id)initFromPtr: (void*)pkt own:(bool)own;
- (id)init: (id)data;
- (void) dealloc;

+ (BmfPacket*)generateEosPacket;
+ (BmfPacket*)generateEofPacket;

- (void*)ptr;
- (bool)defined;
- (void)setTimestamp: (long)ts;
- (long)timestamp;
- (bool)is: (Class) class_type;
- (id)get: (Class) class_type;

@end
