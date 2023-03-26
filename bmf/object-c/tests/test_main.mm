#import <Foundation/Foundation.h>
#import "test_VideoFrame.h"
#import "test_AudioFrame.h"
#import "test_Packet.h"
#import "test_Task.h"
#import "test_ModuleFunctor.h"

void test_main()
{
    int total = 0;
    int pass = 0;
    
    int result = 0;
    
    @autoreleasepool {
        result = [[BmfVideoFrameTests alloc] testAll];
        NSLog(@"BmfVideoFrameTests    -> %d\n", result);
        pass += result == 0;
        total += 1;
    }


    @autoreleasepool {
        result = [[BmfAudioFrameTests alloc] testAll];
        NSLog(@"BmfAudioFrameTests    -> %d\n", result);
        pass += result == 0;
        total += 1;
    }
    

    @autoreleasepool {
        result = [[BmfPacketTests alloc] testAll];
        NSLog(@"BmfPacketTests    -> %d\n", result);
        pass += result == 0;
        total += 1;
    }


    @autoreleasepool {
        result = [[BmfTaskTests alloc] testAll];
        NSLog(@"BmfTaskTests    -> %d\n", result);
        pass += result == 0;
        total += 1;
    }

    
//    result = [[BmfModuleFunctorTests alloc] testAll];
//    NSLog(@"BmfModuleFunctorTests    -> %d\n", result);
//    pass += result == 0;
//    total += 1;

    NSLog(@"Total:%d, Pass:%d, Fail:%d\n", total, pass, total - pass);
    
}

int main(int argc, const char * argv[]) 
{
    for(int i = 0; i < 100000; ++i){ //NOTE: loop to test memleaks
        NSLog(@"=================== %03d ======================\n", i);
        test_main();
        NSLog(@"\n\n");
    }
    
    return 0;
}
