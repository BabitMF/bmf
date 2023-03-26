#import <bmf/oc/Rational.h>

@interface BmfRational()
@property (nonatomic, assign) int num;
@property (nonatomic, assign) int den;
@end

@implementation BmfRational

- (id)init: (int)num den:(int)den{
    if((self = [super init]) != nil){
        self.num = num;
        self.den = den;
    }
    return self;
}

- (int)num{
    return _num;
}

- (int)den{
    return _den;
}

@end
