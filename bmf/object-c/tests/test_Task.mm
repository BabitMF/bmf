#import "test_Task.h"
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
#import "bmf/oc/Task.h"

@implementation BmfTaskTests

- (int) testAll
{
    NSMutableArray *inputs= [NSMutableArray arrayWithCapacity:2];
    [inputs addObject:[NSNumber numberWithInt:0]];
    NSMutableArray *outputs= [NSMutableArray arrayWithCapacity:2];
    [outputs addObject:[NSNumber numberWithInt:1]];
    [outputs addObject:[NSNumber numberWithInt:2]];
    [outputs addObject:[NSNumber numberWithInt:3]];
    BmfTask *task = [[BmfTask alloc] init:1 istream_ids:inputs ostream_ids:outputs];
    NSMutableArray *input_ids = [task getInputStreamIds];
    NSMutableArray *output_ids = [task getOutputStreamIds];
    if ([input_ids count] != 1 || [output_ids count] != 3){
        return 1;
    }
    
    [task setTimestamp: 10001];
    if ([task timestamp] != 10001) {
        return 2;
    }

    BmfVideoFrame *vf_1 = [[BmfVideoFrame alloc] init:1080 height:720 channels:3 format:HmpChannelFormat::kNHWC dtype:HmpScalarType::kUInt8 device: "cpu" pinned_memory:false];
    [vf_1 setPts:1111];
    BmfPacket* pkt1 = [[BmfPacket alloc] init:vf_1];
    BmfVideoFrame *vf_2 = [[BmfVideoFrame alloc] init:1080 height:720 channels:3 format:HmpChannelFormat::kNHWC dtype:HmpScalarType::kUInt8 device: "cpu" pinned_memory:false];
    [vf_2 setPts:2222];
    BmfPacket* pkt2 = [[BmfPacket alloc] init:vf_2];
    BmfVideoFrame *vf_3 = [[BmfVideoFrame alloc] init:1080 height:720 channels:3 format:HmpChannelFormat::kNHWC dtype:HmpScalarType::kUInt8 device: "cpu" pinned_memory:false];
    [vf_3 setPts:3333];
    BmfPacket* pkt3 = [[BmfPacket alloc] init:vf_3];
    BmfVideoFrame *vf_4 = [[BmfVideoFrame alloc] init:1080 height:720 channels:3 format:HmpChannelFormat::kNHWC dtype:HmpScalarType::kUInt8 device: "cpu" pinned_memory:false];
    [vf_4 setPts:4444];
    BmfPacket* pkt4 = [[BmfPacket alloc] init:vf_4];
    [task fillInputPacket:0 pkt:pkt1];
    [task fillOutputPacket:1 pkt:pkt2];
    [task fillOutputPacket:2 pkt:pkt3];
    [task fillOutputPacket:3 pkt:pkt4];
    BmfPacket *tpkt1 = [task popPacketFromInQueue:0];
    BmfPacket *tpkt2 = [task popPacketFromOutQueue:1];
    BmfPacket *tpkt3 = [task popPacketFromOutQueue:2];
    BmfPacket *tpkt4 = [task popPacketFromOutQueue:3];
    long ts1 = [[tpkt1 get:[BmfVideoFrame class]] pts];
    long ts2 = [[tpkt2 get:[BmfVideoFrame class]] pts];
    long ts3 = [[tpkt3 get:[BmfVideoFrame class]] pts];
    long ts4 = [[tpkt4 get:[BmfVideoFrame class]] pts];
    if (ts1 != 1111 || ts2 != 2222 || ts3 != 3333 || ts4 != 4444) {
        return 3;
    }

    return 0;
}

@end
