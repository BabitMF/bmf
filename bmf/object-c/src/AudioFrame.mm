#import "bmf/oc/AudioFrame.h"
#import "bmf/sdk/audio_frame.h"
#import "bmf/sdk/json_param.h"

using namespace bmf_sdk;
using namespace hmp;

@interface BmfAudioFrame()
@property (nonatomic, assign) AudioFrame* p;
@property (nonatomic, assign) bool own;
@end

@implementation BmfAudioFrame

- (id)initFromPtr:(void*)af own:(bool)own{
    if((self = [super init]) != nil){
        self.p = (AudioFrame*)af;
        self.own = own;
    }
    return self;
}

- (void) dealloc
{
    delete self.p;
}

- (BmfAudioFrame*)init:(int)samples layout:(int)layout planer:(bool)planer dtype:(HmpScalarType)dtype
{
    if((self = [super init]) != nil){
        auto tensor_options = TensorOptions((ScalarType)dtype);
        AudioFrame *af = new AudioFrame(samples, layout, planer, tensor_options);
        self.p = af;
        self.own = true;
    }
    return self;
}

- (BmfAudioFrame*)init:(NSMutableArray*)data size:(int)size layout:(int)layout planer:(bool)planer
{
    if((self = [super init]) != nil){
        Tensor tensor_array[size];
        int i=0;
        for(HmpTensor *hmp_tensor in data){
            Tensor *tensor = (Tensor*)[hmp_tensor ptr];
            tensor_array[i] = *tensor;
            i++;
        }
        std::vector<Tensor> tensor_list(tensor_array, tensor_array+size);
        AudioFrame *af = new AudioFrame(tensor_list, layout, planer);
        self.p = af;
        self.own = true;
    }
    return self;
}

- (bool)defined{
    return (bool)self.p;
}

- (void*)ptr{
    return (void*)self.p;
}

- (HmpScalarType)dtype{
    return (HmpScalarType)self.p->dtype();
}

- (bool)planer{
    return self.p->planer();
}

- (int)nsamples{
    return self.p->nsamples();
}

- (int)nchannels{
    return self.p->nchannels();
}

- (void)setSampleRate:(float)sr{
    self.p->set_sample_rate(sr);
}

- (float)sampleRate{
    return self.p->sample_rate();
}

- (NSMutableArray*)planes{
    NSMutableArray *planes= [NSMutableArray arrayWithCapacity:2];
    auto len = self.p->planes().size();
    for(int i = 0; i < len; ++i){
        Tensor *tensor = new Tensor(self.p->plane(i));
        HmpTensor *hmp_tensor = [[HmpTensor alloc]initFromPtr:tensor :true];
        [planes addObject:hmp_tensor];
    }
    return planes;
}

- (int)nplanes{
    return self.p->nplanes();
}

- (HmpTensor*)plane:(int)i{
    Tensor *tensor = new Tensor(self.p->plane(i));
    return [[HmpTensor alloc]initFromPtr:tensor :true];
}


- (void)copyProps: (BmfAudioFrame*)from{
    AudioFrame *a = (AudioFrame*)[from ptr];
    self.p->copy_props(*a);
}

- (void)privateMerge: (BmfAudioFrame*)af{
    AudioFrame *from = (AudioFrame*)[af ptr];
    self.p->private_merge(*from);
}

- (id)privateGet: (BmfOpaqueDataKey)key{
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

- (void)privateAttach: (BmfOpaqueDataKey)key option:(id)option{
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

@end
