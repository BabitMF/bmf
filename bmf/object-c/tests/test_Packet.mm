#import "test_Packet.h"
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
#import "bmf/oc/Packet.h"

@implementation BmfPacketTests

- (int) testAll
{
    // EOF & EOS
    BmfPacket *eof_pkt = [BmfPacket generateEofPacket];
    BmfPacket *eos_pkt = [BmfPacket generateEosPacket];
    if ([eof_pkt timestamp] != 9223372036854775804 || [eos_pkt timestamp] != 9223372036854775805) {
        return 1;
    }

    // construct from videoframe
    BmfVideoFrame *vf = [[BmfVideoFrame alloc] init:1080 height:720 channels:3 format:HmpChannelFormat::kNHWC dtype:HmpScalarType::kUInt8 device:"cpu" pinned_memory:false];
    [vf setPts:1001];
    BmfPacket *pkt = [[BmfPacket alloc]init:vf];
    [pkt setTimestamp:303];
    if ([pkt defined]!=1 || [pkt timestamp]!=303 || [pkt is:[BmfVideoFrame class]]!=1) {
        return 2;
    }
    BmfVideoFrame *bvf = [pkt get:[BmfVideoFrame class]];
    if ([bvf pts] != 1001) {
        return 3;
    }
    
    // construct from audioframe
    BmfAudioFrame *af = [[BmfAudioFrame alloc] init:2 layout:1 planer:true dtype:HmpScalarType::kUInt8];
    BmfPacket *pkt2 = [[BmfPacket alloc]init:af];
    if ([pkt2 is:[BmfAudioFrame class]] != 1) {
        return 4;
    }
    BmfAudioFrame *baf = [pkt2 get:[BmfAudioFrame class]];
    
    // construct from json
    char* ch = "{\"name\":\"ios_sdk\",\"option\":{\"path\":\"my_path\",\"entry\":\"my_entry\"}}";
    NSString *string = [[NSString alloc] initWithCString:ch encoding:NSUTF8StringEncoding];
    NSError *error = nil;
    NSData *data= [string dataUsingEncoding:NSUTF8StringEncoding];
    id jsonObject = [NSJSONSerialization JSONObjectWithData:data options:NSJSONReadingAllowFragments error:&error];
    BmfPacket *pkt3 = [[BmfPacket alloc]init:jsonObject];
    id outObject = [pkt3 get:[NSDictionary class]];
    if ([pkt3 is:[NSDictionary class]] != 1 || [outObject isEqual: jsonObject] != 1){
        return 5;
    }

    return 0;
}

@end
