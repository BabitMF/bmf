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

#include <bmf_lite.h>
#include <string>

#import "BmfLiteDemoCore.h"
#import "BmfLiteDemoSdInterface.h"
#import "SdViewController.h"
#import <AVFoundation/AVFoundation.h>
#import <Foundation/Foundation.h>

#import "bmf_demo-Bridging-Header.h"
#import "bmf_demo-Swift.h"

@interface SdViewController () <UITextFieldDelegate> {
  NSString *text;
}

@property(nonatomic, strong) IBOutlet UIImageView *imageView;

@property(nonatomic, strong) IBOutlet UIButton *generateButton;

@property(nonatomic) NSTimer *timer;

@property(nonatomic, assign) double progressValue;

@property(nonatomic, strong) IBOutlet UIProgressView *progressView;

//@property (nonatomic, retain) SDExcutor* excutor;

@property(nonatomic, strong) UILabel *textLabel;

@property(nonatomic, strong) UIImage *image;

@property(nonatomic, assign) uint32_t seeds;

@property(nonatomic, assign) double steps;

@property(nonatomic, assign) double interval;

@property(nonatomic, strong) UILabel *interval_label;

@property(nonatomic) NSTimer *result_timer;

@end

@implementation SdViewController {
  bmf_lite::IAlgorithm *sd_excutor_;
}

- (void)viewDidLoad {
  [super viewDidLoad];

  int w = self.view.bounds.size.width;
  int h = self.view.bounds.size.height / 2;
  self.imageView = [[UIImageView alloc] initWithFrame:CGRectMake(0, 200, w, h)];

  [self.view addSubview:self.imageView];

  self.progressView = [[UIProgressView alloc]
      initWithProgressViewStyle:UIProgressViewStyleDefault];
  self.progressView.frame = CGRectMake(0, h + 250, w, 50);
  [self.view addSubview:self.progressView];

  self.textLabel = [[UILabel alloc] init];
  self.textLabel.text = @"Downloading";
  self.textLabel.font = [UIFont systemFontOfSize:16];
  self.textLabel.textColor = UIColor.blueColor;
  self.textLabel.frame = CGRectMake(0, h + 200, w, 50);
  [self.view addSubview:self.textLabel];

  BMFLITE_MODEL_REGIST(6, bmf_lite::demo::BMFLiteDemoSdInterface);

  sd_excutor_ = bmf_lite::AlgorithmFactory::createAlgorithmInterface();
  bmf_lite::Param init_param;
  init_param.setInt("change_mode", 2);
  init_param.setString("instance_id", "sd1");
  init_param.setInt("algorithm_type", 6);
  init_param.setInt("algorithm_version", 0);

  assert(sd_excutor_->setParam(init_param) == 0);

  bmf_lite::Param set_param;
  set_param.setInt("change_mode", 5);
  set_param.setString("instance_id", "sd1");
  set_param.setInt("algorithm_type", 6);
  set_param.setInt("algorithm_version", 0);
  assert(sd_excutor_->setParam(set_param) == 0);
  self.timer = [NSTimer scheduledTimerWithTimeInterval:1.0
                                                target:self
                                              selector:@selector(updateProgress)
                                              userInfo:nil
                                               repeats:YES];
}

- (void)getResut {
  bmf_lite::VideoFrame oframe;
  bmf_lite::Param output_param;
  output_param.setInt("request_mode", 2);
  assert(sd_excutor_->getVideoFrameOutput(oframe, output_param) == 0);
  int completed = 0;
  output_param.getInt("process_status", completed);
  if (completed) {
    NSLog(@"process completed");
    std::shared_ptr<bmf_lite::VideoBuffer> ibuf = oframe.buffer();
      
    CGImageRef cg_img = (CGImageRef)ibuf->data();
    NSLog(@"get cgimag completed");
    CFRetain(cg_img);
      
    float original_w = CGImageGetWidth(cg_img);

    float original_h = CGImageGetHeight(cg_img);
    NSLog(@"cgimage width:%f, height:%f", original_w, original_h);
    CGSize original_size = CGSizeMake(original_w, original_h);

    CGFloat target_width = self.imageView.bounds.size.width;
    CGFloat target_height = (original_size.height / original_size.width) * target_width;
    CGSize target_size = CGSizeMake(target_width, target_height);

    CGContextRef context = CGBitmapContextCreate(NULL, target_size.width, target_size.height,
                                                CGImageGetBitsPerComponent(cg_img), 4 * target_size.width, CGImageGetColorSpace(cg_img), kCGImageAlphaPremultipliedLast);

    CGRect rect = CGRectMake(0, 0, target_size.width, target_size.height);
    CGContextDrawImage(context, rect, cg_img);
    CGImageRef resized_image = CGBitmapContextCreateImage(context);
    CGContextRelease(context);
    self.imageView.image = [UIImage imageWithCGImage:resized_image];
    CGImageRelease(resized_image);
    CFRelease(cg_img);
    [self.result_timer invalidate];
    self.result_timer = nil;
    self.textLabel.text = @"process success!";
  }
}

- (void)buttonClicked:(UIButton *)button {
  if (text == nil) {
    return;
  }
  if (self.textField != nil) {
      [self.textField resignFirstResponder];
  }
  std::string input_text = std::string([text UTF8String]);
  NSLog(@"positive_prompt:%@",text);
  bmf_lite::Param input_param;
  input_param.setString("positive_prompt", input_text);
  self.textLabel.text = @"processing...";
  bmf_lite::VideoFrame iframe;
  sd_excutor_->processVideoFrame(iframe, input_param);
  self.result_timer =
      [NSTimer scheduledTimerWithTimeInterval:0.5
                                       target:self
                                     selector:@selector(getResut)
                                     userInfo:nil
                                      repeats:YES];
}

- (void)updateProgress {
  bmf_lite::Param output_param;
  bmf_lite::VideoFrame oframe;
  output_param.setInt("request_mode", 1);
  sd_excutor_->getVideoFrameOutput(oframe, output_param);

  double progress_value;
  output_param.getDouble("progress_value", progress_value);
  int status = 0;
  output_param.getInt("init_status", status);

  if (status == 0) {
    self.textLabel.text = @"Downloading model";
    [self.progressView setProgress:progress_value animated:YES];
  } else if (status == 1) {
    self.textLabel.text = @"Uncompressing model";
    [self.progressView removeFromSuperview];
  } else if (status == 2) {
    self.textLabel.text = @"Loading model";
    [self.progressView removeFromSuperview];
  } else if (status == 3) {
    [self.timer invalidate];
    self.timer = nil;
    self.textField =
        [[UITextField alloc] initWithFrame:CGRectMake(10, 100, 200, 30)];
    self.textField.borderStyle = UITextBorderStyleRoundedRect;
    self.textField.layer.borderWidth = 1.0;
    self.textField.delegate = self;
    [self.view addSubview:self.textField];

    self.generateButton = [UIButton buttonWithType:UIButtonTypeRoundedRect];
    self.generateButton.frame = CGRectMake(260, 100, 100, 30);
    self.generateButton.backgroundColor = UIColor.blueColor;
    [self.generateButton setTitle:@"generate" forState:UIControlStateNormal];
    [self.generateButton addTarget:self
                            action:@selector(buttonClicked:)
                  forControlEvents:UIControlEventTouchUpInside];
    [self.view addSubview:self.generateButton];
    self.textLabel.text = @"Loading model completed!";
  }
}

- (BOOL)textFieldShouldReturn:(UITextField *)textField {
  [textField resignFirstResponder];
  text = textField.text;
  return YES;
}

- (void)drawImage:(nonnull UIImage *)image {
}

@end
