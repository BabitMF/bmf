#import "bmf/oc/Packet.h"
#import "bmf/sdk/packet.h"
#import "bmf/sdk/video_frame.h"
#import "bmf/sdk/audio_frame.h"
#import "bmf/sdk/json_param.h"

using namespace bmf_sdk;
using namespace hmp;

@interface BmfPacket()
@property (nonatomic, assign) Packet* p;
@property (nonatomic, assign) bool own;
@end

@implementation BmfPacket

- (id)initFromPtr: (void*)pkt own:(bool)own{
    if((self = [super init]) != nil){
        self.p = (Packet*)pkt;
        self.own = own;
    }
    return self;
}

- (id)init: (id)data{
    if((self = [super init]) != nil){
        if (data == nil){
            self.p = 0;
            self.own = false;
        }else if([data isKindOfClass:[BmfVideoFrame class]]){
            VideoFrame* vf = (VideoFrame*)[data ptr];
            Packet *pkt = new Packet(*vf);
            self.p = pkt;
            self.own = true;
        }else if([data isKindOfClass:[BmfAudioFrame class]]){
            AudioFrame* af = (AudioFrame*)[data ptr];
            Packet *pkt = new Packet(*af);
            self.p = pkt;
            self.own = true;
        }else{
            if([NSJSONSerialization isValidJSONObject:data]){
                NSError *error = nil;
                NSData *data_str = [NSJSONSerialization dataWithJSONObject:data options:NSJSONReadingAllowFragments error:&error];
                NSString *ns_string = [[NSString alloc]initWithData:data_str encoding:NSUTF8StringEncoding];
                const char *str =[ns_string UTF8String];
                JsonParam *j = new JsonParam();
                j->parse(str);
                self.p = new Packet(*j);
                self.own = true;
            }else{
                NSLog(@"data format not supported.");
            }
        }
    }
    return self;
}

+ (BmfPacket*)generateEosPacket{
    Packet *pkt= new Packet(Packet::generate_eos_packet());
    return [[BmfPacket alloc] initFromPtr:pkt own:true];
}

+ (BmfPacket*)generateEofPacket{
    Packet *pkt= new Packet(Packet::generate_eof_packet());
    return [[BmfPacket alloc] initFromPtr:pkt own:true];
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

- (bool)defined{
    return self.p != nullptr && self.p->operator bool();
}

- (void)setTimestamp: (long)ts{
    self.p->set_timestamp(ts);
}

- (long)timestamp{
    return self.p->timestamp();
}

- (bool)is: (Class) class_type{
    if(class_type == [BmfVideoFrame class]){
        return self.p->is<VideoFrame>();
    }else if (class_type == [BmfAudioFrame class]){
        return self.p->is<AudioFrame>();
    }else if (class_type == [NSDictionary class]){
        return self.p->is<JsonParam>();
    }else{
        NSLog(@"data format not supported.");
        return false;
    }
}

- (id)get: (Class) class_type{
    if(class_type == [BmfVideoFrame class]){
        VideoFrame *vf = new VideoFrame(self.p->get<VideoFrame>());
        return [[BmfVideoFrame alloc] initFromPtr:vf own:true];
    }else if (class_type == [BmfAudioFrame class]){
        AudioFrame *af = new AudioFrame(self.p->get<AudioFrame>());
        return [[BmfAudioFrame alloc] initFromPtr:(void*)af own:true];
    }else if(class_type == [NSDictionary class]){
        auto json = std::make_unique<JsonParam>(self.p->get<JsonParam>());
        auto str = json->dump();
        NSString *string = [[NSString alloc] initWithCString:str.c_str() encoding:NSUTF8StringEncoding];
        NSError *error = nil;
        NSData *data= [string dataUsingEncoding:NSUTF8StringEncoding];
        id jsonObject = [NSJSONSerialization JSONObjectWithData:data options:NSJSONReadingAllowFragments error:&error];
        return jsonObject;
    }else{
        NSLog(@"data format not supported.");
        return nil;
    }
}

@end
