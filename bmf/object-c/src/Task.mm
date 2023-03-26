#import "bmf/oc/Task.h"
#import "bmf/sdk/packet.h"
#import "bmf/sdk/video_frame.h"
#import "bmf/sdk/task.h"

using namespace bmf_sdk;
using namespace hmp;

@interface BmfTask()
@property (nonatomic, assign) Task* p;
@property (nonatomic, assign) bool own;
@end

@implementation BmfTask

- (id)initFromPtr: (void*)task own:(bool)own{
    if((self = [super init]) != nil){
        self.p = (Task*)task;
        self.own = own;
    }
    return self;
}

- (id)init: (int)node_id istream_ids:(NSMutableArray*)istream_ids ostream_ids:(NSMutableArray*)ostream_ids{
    if((self = [super init]) != nil){
        long in_len = [istream_ids count];
        long out_len = [ostream_ids count];

        int inputs[in_len];
        int outputs[out_len];
        int i=0,j=0;
        for(NSNumber *obj in istream_ids){
            inputs[i] = [obj intValue];
            i++;
        }
        for(NSNumber *obj in ostream_ids){
            outputs[j] = [obj intValue];
            j++;
        }
        
        std::vector<int> iids(inputs, inputs + in_len);
        std::vector<int> oids(outputs, outputs + out_len);
        self.p = new Task(node_id, iids, oids);
        self.own = true;
    }
    return self;
}

- (void) dealloc{
    delete self.p;
}

- (void*)ptr{
    return self.p;
}

- (void)setTimestamp: (long)ts{
    self.p->set_timestamp(ts);
}

- (long)timestamp{
    return self.p->timestamp();
}

- (bool)fillInputPacket: (int)stream_id pkt:(BmfPacket*)pkt{
    Packet* p = (Packet*)[pkt ptr];
    int rc = self.p->fill_input_packet(stream_id, *p);
    return rc!=0;
}

- (bool)fillOutputPacket: (int)stream_id pkt:(BmfPacket*)pkt{
    Packet* p = (Packet*)[pkt ptr];
    int rc = self.p->fill_output_packet(stream_id, *p);
    return rc!=0;
}

- (BmfPacket*)popPacketFromOutQueue: (int)stream_id{
    Packet p;
    if(self.p->pop_packet_from_out_queue(stream_id, p)){
        Packet *pkt = new Packet(p);
        return [[BmfPacket alloc]initFromPtr: pkt own:true];
    }
    else{
        throw std::runtime_error(
            fmt::format("stream id out of range or no packet to pop from output stream {}", stream_id));
    }
}

- (BmfPacket*)popPacketFromInQueue: (int)stream_id{
    Packet p;
    if(self.p->pop_packet_from_input_queue(stream_id, p)){
        Packet *pkt = new Packet(p);
        return [[BmfPacket alloc]initFromPtr: pkt own:true];
    }
    else{
        throw std::runtime_error(
            fmt::format("stream id out of range or no packet to pop from output stream {}", stream_id));
    }
}

- (NSMutableArray*)getInputStreamIds{
    auto ids = self.p->get_input_stream_ids();
    auto n = ids.size();
    NSMutableArray *array = [NSMutableArray arrayWithCapacity:2];
    for(int i=0; i<n; i++){
        NSNumber* number = [NSNumber numberWithInt: ids[i]];
        [array addObject:number];
    }
    return array;
}

- (NSMutableArray*)getOutputStreamIds{
    auto ids = self.p->get_output_stream_ids();
    auto n = ids.size();
    NSMutableArray *array = [NSMutableArray arrayWithCapacity:2];
    for(int i=0; i<n; i++){
        NSNumber* number = [NSNumber numberWithInt: ids[i]];
        [array addObject:number];
    }
    return array;
}

@end

