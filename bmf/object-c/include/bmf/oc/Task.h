#import <Foundation/Foundation.h>
#import <bmf/oc/VideoFrame.h>
#import <bmf/oc/Packet.h>

@interface BmfTask: NSObject

- (id)initFromPtr: (void*)task own:(bool)own;
- (id)init: (int)node_id istream_ids:(NSMutableArray*)istream_ids ostream_ids:(NSMutableArray*)ostream_ids;
- (void) dealloc;

- (void*)ptr;
- (void)setTimestamp: (long)ts;
- (long)timestamp;
- (bool)fillInputPacket: (int)stream_id pkt:(BmfPacket*)pkt;
- (bool)fillOutputPacket: (int)stream_id pkt:(BmfPacket*)pkt;
- (BmfPacket*)popPacketFromOutQueue: (int)stream_id;
- (BmfPacket*)popPacketFromInQueue: (int)stream_id;
- (NSMutableArray*)getInputStreamIds;
- (NSMutableArray*)getOutputStreamIds;

@end
