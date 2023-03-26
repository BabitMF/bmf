#include <vector>
#import <hmp/oc/Tensor.h>
#import <test_Tensor.h>
#include <hmp/core/scalar.h>

@implementation BmfTensorTests

- (int) testAll
{
     @autoreleasepool{
        NSMutableArray *shape = [NSMutableArray arrayWithCapacity: 3];
        [shape addObject : [NSNumber numberWithLong : 2]];
        [shape addObject : [NSNumber numberWithLong : 3]];
        [shape addObject : [NSNumber numberWithLong : 4]];

        HmpTensor *t0 = [HmpTensor empty: shape DType:kInt32 Device:@"cpu" Pinned:false];
        if([t0 size : 0] != 2 || [t0 size : 1] != 3 || [t0 size : 2] != 4){
            return 1;
        }
        if(![t0 defined]){
            return 2;
        }


        HmpTensor *t1 = [t0 clone];
        if([t1 size : 0] != 2 || [t1 size : 1] != 3 || [t1 size : 2] != 4){
            return 3;
        }

        HmpTensor *t2 = [t0 alias];
        if([t2 size : 0] != 2 || [t2 size : 1] != 3 || [t2 size : 2] != 4){
            return 4;
        }

        //view
        NSMutableArray *shape2 = [NSMutableArray arrayWithCapacity: 3];
        [shape2 addObject : [NSNumber numberWithLong : 4]];
        [shape2 addObject : [NSNumber numberWithLong : 3]];
        [shape2 addObject : [NSNumber numberWithLong : 2]];
        HmpTensor *t3 = [t2 view : shape2];
        if([t3 size : 0] != 4 || [t3 size : 1] != 3 || [t3 size : 2] != 2){
            return 5;
        }

        HmpTensor *t4 = [t2 reshape : shape2];
        if([t4 size : 0] != 4 || [t4 size : 1] != 3 || [t4 size : 2] != 2){
            return 6;
        }

        // permute
        NSMutableArray *dims = [NSMutableArray arrayWithCapacity: 3];
        [dims addObject : [NSNumber numberWithLong : 2]];
        [dims addObject : [NSNumber numberWithLong : 1]];
        [dims addObject : [NSNumber numberWithLong : 0]];
        HmpTensor *t5 = [t2 permute : dims];
        if([t5 size : 0] != 4 || [t5 size : 1] != 3 || [t5 size : 2] != 2){
            return 10;
        }
        if([t5 is_contiguous]){
            return 11;
        }

        // slice & select
        HmpTensor *t6 = [t0 slice : -1 : 1 : 4 : 2];
        if([t6 dim] != 3 || [t6 size : 0] != 2 || [t6 size : 1] != 3 || [t6 size : 2] != 2){
            return 20;
        }
        if([t6 is_contiguous]){
            return 21;
        }

        HmpTensor *t7 = [t0 select : 1 : 1];
        if([t7 dim] != 2 || [t7 size : 0] != 2 || [t7 size : 1] != 4){
            return 22;
        }

        //
        if([t0 stride : 0] != 12 || [t0 stride : 1] != 4 || [t0 stride : 2] != 1){
            return 30;
        }
        if([t0 nbytes] != [t0 itemsize] * [t0 nitems] || [t0 nitems] != 24){
            return 31;
        }
        if([t0 unsafe_data] == 0){
            return 32;
        }


        //
        std::vector<float> data(24);
        HmpTensor *t8 = [HmpTensor from_buffer : data.data() : shape2 : kFloat32 : @"cpu" : nullptr];
        if([t8 dim] != 3 || [t8 size : 0] != 4 || [t8 size : 1] != 3 || [t8 size : 2] != 2){
            return 40;
        }

    }

    return 0;

}

@end
