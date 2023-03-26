#import <Foundation/Foundation.h>
#import <bmf/oc/VideoFrame.h>
#import <bmf/oc/Packet.h>
#import <bmf/oc/Task.h>

@interface BmfModuleFunctor: NSObject

- (id)initFromPtr: (void*)mf own:(bool)own;

- (id)init: (char *)name type:(char *)type path:(char*)path entry:(char *)entry option:(id)option ninputs:(int)ninputs noutputs:(int)noutputs;

- (void) dealloc;

- (void*)ptr;

@end
