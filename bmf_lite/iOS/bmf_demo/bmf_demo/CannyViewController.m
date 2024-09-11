/*
 * Copyright 2024 Babit Authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#import "CannyViewController.h"
#import "BmfLiteDemoCore.h"
#import <AVFoundation/AVFoundation.h>
#import <Foundation/Foundation.h>

@interface CannyViewController ()<AVCaptureVideoDataOutputSampleBufferDelegate>
{
    MTKView *_view;
    BmfLiteDemoPlayer *_player;
    AVCaptureSession *session;
}

@end

@implementation CannyViewController

- (IBAction)switchValueChanged:(id)sender {
    if (self.camera_switch.isOn) {
        [self setupAVCapture:YES];
    } else {
        [self setupAVCapture:NO];
    }
}

- (void)viewDidLoad {
    [super viewDidLoad];
    session = nil;
    // Do any additional setup after loading the view.
//    _view = (MTKView *)self.view;
    CGRect rect = CGRectMake(0, 0, self.view.bounds.size.width, self.view.bounds.size.height);
    _view = [[MTKView alloc] initWithFrame:rect device:MTLCreateSystemDefaultDevice()];
    [self.view addSubview:_view];
    self.camera_switch = [[UISwitch alloc]initWithFrame:CGRectMake(0, 0, 200, 100)];
    self.camera_switch.on = YES;
    [self.camera_switch addTarget:self action:@selector(switchValueChanged:) forControlEvents:UIControlEventValueChanged];
    [self.view addSubview:self.camera_switch];
    [self setupAVCapture:TRUE];
}

- (void)captureOutput:(AVCaptureOutput *)captureOutput
didOutputSampleBuffer:(CMSampleBufferRef)sampleBuffer
       fromConnection:(AVCaptureConnection *)connection
{
    [_player consume:sampleBuffer : NO];
}

- (void)setupAVCapture:(BOOL) front_camera
{
    if (front_camera) {
        _player = [[BmfLiteDemoPlayer alloc] initWithMTKView:_view : 1 : BmfLiteDemoAlgoType::BMFLITE_DEMO_ALGO_CANNY];
    } else {
        _player = [[BmfLiteDemoPlayer alloc] initWithMTKView:_view : 2 : BmfLiteDemoAlgoType::BMFLITE_DEMO_ALGO_CANNY];
    }

    NSError *error = nil;
    if (session != nil) {
        [session stopRunning];
    }

    session = [AVCaptureSession new];
    if ([[UIDevice currentDevice] userInterfaceIdiom] == UIUserInterfaceIdiomPhone)
//        [session setSessionPreset:AVCaptureSessionPreset640x480];
        [session setSessionPreset:AVCaptureSessionPreset1280x720];
    
    else
        [session setSessionPreset:AVCaptureSessionPresetPhoto];

    AVCaptureDevice* device;
    if (front_camera) {
        device = [AVCaptureDevice defaultDeviceWithDeviceType:AVCaptureDeviceTypeBuiltInWideAngleCamera mediaType:AVMediaTypeVideo position:AVCaptureDevicePositionFront];
    } else {
        device = [AVCaptureDevice defaultDeviceWithDeviceType:AVCaptureDeviceTypeBuiltInWideAngleCamera mediaType:AVMediaTypeVideo position:AVCaptureDevicePositionBack];
    }
    AVCaptureDeviceInput *deviceInput = [AVCaptureDeviceInput deviceInputWithDevice:device error:&error];
    //    isUsingFrontFacingCamera = NO;
    if ([session canAddInput:deviceInput]){
        [session addInput:deviceInput];
    }

    AVCaptureVideoDataOutput* videoOutput = [[AVCaptureVideoDataOutput alloc] init];
    videoOutput.alwaysDiscardsLateVideoFrames = YES;
    videoOutput.videoSettings = [NSDictionary dictionaryWithObject:[NSNumber numberWithInt:kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange] forKey:(id)kCVPixelBufferPixelFormatTypeKey];
    if ([session canAddOutput:videoOutput])
        [session addOutput:videoOutput];

    dispatch_queue_t queue = dispatch_queue_create("myQueue", NULL);
    [videoOutput setSampleBufferDelegate:self queue:queue];
    AVCaptureVideoPreviewLayer* preLayer = [AVCaptureVideoPreviewLayer layerWithSession: session];
//    preLayer.frame = CGRectMake(0, 0, 100,100);
    preLayer.videoGravity = AVLayerVideoGravityResizeAspectFill;
    [self.view.layer addSublayer:preLayer];

    [session startRunning];
}

@end
