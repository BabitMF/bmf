#import <Foundation/Foundation.h>
#import <bmf/oc/Rational.h>
#import "bmf/oc/OpaqueDataKey.h"
#import "hmp/oc/Formats.h"
#import <hmp/oc/Device.h>
#import <hmp/oc/Scalar.h>
#import <hmp/oc/ScalarType.h>
#import <hmp/oc/Tensor.h>

@interface BmfAudioFrame : NSObject

- (id)initFromPtr:(void*)af own:(bool)own;
- (void) dealloc;

- (BmfAudioFrame*)init:(int)samples layout:(int)layout planer:(bool)planer dtype:(HmpScalarType)dtype;
- (BmfAudioFrame*)init:(NSMutableArray*)data size:(int)size layout:(int)layout planer:(bool)planer;

- (bool)defined;
- (void*)ptr;
- (HmpScalarType)dtype;
- (bool)planer;
- (int)nsamples;
- (int)nchannels;
- (void)setSampleRate:(float)sr;
- (float)sampleRate;
- (NSMutableArray*)planes;
- (int)nplanes;
- (HmpTensor*)plane:(int)i;

- (void)copyProps: (BmfAudioFrame*)from;
- (void)privateMerge: (BmfAudioFrame*)vf;
- (id)privateGet: (BmfOpaqueDataKey)key;
- (void)privateAttach: (BmfOpaqueDataKey)key option:(id)option;
- (void)setPts: (long)pts;
- (long)pts;
- (void)setTimeBase: (BmfRational*)rational;
- (BmfRational*)timeBase;

@end
