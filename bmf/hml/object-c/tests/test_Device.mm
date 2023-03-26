
#import <hmp/oc/Device.h>
#import <test_Device.h>


@implementation BmfDeviceTests

- (int) testAll
{
    //constructors
    @autoreleasepool{
        // default constructors
        HmpDevice *device = [[HmpDevice alloc] init];
        if([device type] != HmpDeviceType::kCPU || [device index] != 0){
            return 1;
        }

        // construct from string
        device = [[HmpDevice alloc] initFromString: @"cpu"];
        if([device type] != HmpDeviceType::kCPU || [device index] != 0){
            return 2;
        }

        if(![[device description] isEqual : @"cpu"]){
            return 3;
        }
    }

    return 0;
}

@end
