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

#import "PlayViewController.h"
#import "BmfLiteDemoCore.h"

@interface PlayViewController ()
{
    MTKView *_view;
    BmfLiteDemoVodPlayer *_player;
}
@end

@implementation PlayViewController

- (void)viewDidLoad {
    [super viewDidLoad];
    // Do any additional setup after loading the view.
    
//    _view = [[MTKView alloc] initWithFrame:self.view.bounds device:MTLCreateSystemDefaultDevice()];
//    [self.view addSubview:_view];

    _view = (MTKView *)self.view;
    _view.device = MTLCreateSystemDefaultDevice();
    

    NSString * video_path = [[NSBundle mainBundle]pathForResource: @"test" ofType:@"mp4"];
    if (video_path == nil) {
        NSLog(@"cannot find test video!");
        return;
    }
    dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), ^{
        self->_player = [[BmfLiteDemoVodPlayer alloc] initWithMTKView:self->_view AndVideoPath:video_path WhetherPlayAudio:YES Compare:YES];
        [_player run];
    });

}

- (void)viewDidDisappear:(BOOL)animated
{
    [self->_player stop];
}

- (IBAction)sliderValueChange:(id)sender{
    NSInteger tag = [sender tag];
    UISlider* ui_slider = (UISlider*)[self.view viewWithTag:tag];
    float value = [ui_slider value];
    [self->_player setSliderValue:value];
}


/*
#pragma mark - Navigation

// In a storyboard-based application, you will often want to do a little preparation before navigation
- (void)prepareForSegue:(UIStoryboardSegue *)segue sender:(id)sender {
    // Get the new view controller using [segue destinationViewController].
    // Pass the selected object to the new view controller.
}
*/

@end
