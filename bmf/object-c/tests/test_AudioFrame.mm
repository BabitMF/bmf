#import "test_AudioFrame.h"
#import <hmp/oc/Device.h>
#import "hmp/oc/Formats.h"
#import <hmp/oc/ScalarType.h>
#import <hmp/oc/Tensor.h>
#import "bmf/oc/OpaqueDataKey.h"
#import <bmf/oc/Rational.h>
#import "bmf/oc/AudioFrame.h"

@implementation BmfAudioFrameTests

- (int) testAll
{
    // default constructors
    BmfAudioFrame *af0 = [[BmfAudioFrame alloc]init];
    if ([af0 defined] != 0) {
        return 1;
    }
    
    // construct from samples & layout
    BmfAudioFrame *af1 = [[BmfAudioFrame alloc] init:2 layout:1 planer:true dtype:HmpScalarType::kUInt8];
    if ([af1 defined] != 1 || [af1 dtype] != HmpScalarType::kUInt8 || [af1 planer] != true || [af1 nsamples] != 2 || [af1 nchannels] != 1 || [af1 sampleRate] != 1 || [af1 nplanes] != 1) {
        return 2;
    }
    [af1 setSampleRate:2];
    [af1 setPts:101];
    BmfRational *rational = [[BmfRational alloc] init:1 den:100];
    [af1 setTimeBase:rational];
    if ([af1 sampleRate] != 2 || [af1 pts] != 101 || [[af1 timeBase] num] != 1 || [[af1 timeBase] den] != 100) {
        return 3;
    }
    HmpTensor *tensor = [af1 plane:0];
    if ([tensor defined] != true) {
        return 4;
    }
    
    // construct from data
    NSMutableArray *data = [af1 planes];
    int size = [af1 nplanes];
    BmfAudioFrame *af2 = [[BmfAudioFrame alloc]init:data size:size layout:1 planer:true];
    if ([af2 defined] != 1 || [af2 dtype] != HmpScalarType::kUInt8 || [af2 planer] != true || [af2 nsamples] != 2 || [af2 nchannels] != 1 || [af2 sampleRate] != 1 || [af2 nplanes] != 1) {
        return 5;
    }
                
    // copyProps
    [af2 copyProps:af1];
    if ([af2 pts] != 101 || [[af2 timeBase] num] != 1 || [[af2 timeBase] den] != 100) {
        return 6;
    }

    // private attach & private get
    char* ch = "{\"name\":\"ios_sdk\",\"option\":{\"path\":\"my_path\",\"entry\":\"my_entry\"}}";
    NSString *string = [[NSString alloc] initWithCString:ch encoding:NSUTF8StringEncoding];
    NSError *error = nil;
    NSData *data_str= [string dataUsingEncoding:NSUTF8StringEncoding];
    id jsonObject = [NSJSONSerialization JSONObjectWithData:data_str options:NSJSONReadingAllowFragments error:&error];
    [af1 privateAttach:kJsonParam option:jsonObject];
    id obj = [af1 privateGet:kJsonParam];
    NSError *error2 = nil;
    NSData *data_str2 = [NSJSONSerialization dataWithJSONObject:obj options:NSJSONReadingAllowFragments error:&error2];
    NSString *ns_string = [[NSString alloc]initWithData:data_str2 encoding:NSUTF8StringEncoding];
    const char *str =[ns_string UTF8String];
    if ([jsonObject isEqual: obj] != 1){
        return 7;
    }

    // private merge
    [af2 privateMerge:af1];
    id obj2 = [af2 privateGet:kJsonParam];
    if ([jsonObject isEqual: obj2] != 1){
        return 8;
    }
    
    return 0;
}

@end
