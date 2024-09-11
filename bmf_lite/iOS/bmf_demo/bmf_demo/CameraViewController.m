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

#import "CameraViewController.h"
#import "BmfLiteDemoCore.h"
#import <AVFoundation/AVFoundation.h>
#import <Foundation/Foundation.h>

@interface CameraViewController ()<AVCaptureVideoDataOutputSampleBufferDelegate>
{
    MTKView *_view;
    BmfLiteDemoPlayer *_player;
    AVCaptureSession *session;
}

@end

@implementation CameraViewController

- (void)viewDidLoad {
    [super viewDidLoad];
    session = nil;
    // Do any additional setup after loading the view.
    _view = (MTKView *)self.view;
    _view.device = MTLCreateSystemDefaultDevice();
    [self setupAVCapture:TRUE];
}

- (void)captureOutput:(AVCaptureOutput *)captureOutput
didOutputSampleBuffer:(CMSampleBufferRef)sampleBuffer
       fromConnection:(AVCaptureConnection *)connection
{
    [_player consume:sampleBuffer :YES];
}
- (IBAction)toggleCamera:(id)sender {
    UISwitch* control = (UISwitch*) sender;
     
    BOOL use_front = [control isOn];
    [self setupAVCapture:use_front];
}

/**
 *  相机初始化方法
 */
- (void)setupAVCapture:(BOOL) front_camera
{
    if (front_camera) {
        _player = [[BmfLiteDemoPlayer alloc] initWithMTKView:_view : 1 : BmfLiteDemoAlgoType::BMFLITE_DEMO_ALGO_DENOISE];
    } else {
        _player = [[BmfLiteDemoPlayer alloc] initWithMTKView:_view : 2 : BmfLiteDemoAlgoType::BMFLITE_DEMO_ALGO_DENOISE];
    }

    NSError *error = nil;
    if (session != nil) {
        [session stopRunning];
    }
    
    // 1 创建session
    session = [AVCaptureSession new];
    // 2 设置session显示分辨率
    if ([[UIDevice currentDevice] userInterfaceIdiom] == UIUserInterfaceIdiomPhone)
//        [session setSessionPreset:AVCaptureSessionPreset640x480];
        [session setSessionPreset:AVCaptureSessionPreset1280x720];
    
    else
        [session setSessionPreset:AVCaptureSessionPresetPhoto];
    
    
    // 3 获取摄像头device,并且默认使用的后置摄像头,并且将摄像头加入到captureSession中
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
    // 4 创建预览output,设置预览videosetting,然后设置预览delegate使用的回调线程,将该预览output加入到session
    AVCaptureVideoDataOutput* videoOutput = [[AVCaptureVideoDataOutput alloc] init];
    videoOutput.alwaysDiscardsLateVideoFrames = YES;
    videoOutput.videoSettings = [NSDictionary dictionaryWithObject:[NSNumber numberWithInt:kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange] forKey:(id)kCVPixelBufferPixelFormatTypeKey];//设置像素格式
    if ([session canAddOutput:videoOutput])
        [session addOutput:videoOutput];
    //    5 显示捕捉画面
    dispatch_queue_t queue = dispatch_queue_create("myQueue", NULL);
    [videoOutput setSampleBufferDelegate:self queue:queue];
    AVCaptureVideoPreviewLayer* preLayer = [AVCaptureVideoPreviewLayer layerWithSession: session];//相机拍摄预览图层
//    preLayer.frame = CGRectMake(0, 0, 100,100);
    preLayer.videoGravity = AVLayerVideoGravityResizeAspectFill;
    [self.view.layer addSublayer:preLayer];
    
    // 6 启动session,output开始接受samplebuffer回调
    [session startRunning];
}

/*
#pragma mark - Navigation

// In a storyboard-based application, you will often want to do a little preparation before navigation
- (void)prepareForSegue:(UIStoryboardSegue *)segue sender:(id)sender {
    // Get the new view controller using [segue destinationViewController].
    // Pass the selected object to the new view controller.
}
*/

- (IBAction)sliderValueChange:(id)sender{
    NSInteger tag = [sender tag];
    UISlider* ui_slider = (UISlider*)[self.view viewWithTag:tag];
    float value = [ui_slider value];
    [self->_player setSliderValue:value];
}

@end
