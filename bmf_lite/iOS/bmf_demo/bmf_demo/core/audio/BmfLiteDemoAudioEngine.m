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

#import "BmfLiteDemoAudioEngine.h"

#pragma mark BmfLiteDemoAudioEngine class extensions

@interface BmfLiteDemoAudioEngine() {
    AVAudioEngine      *_engine;
    AVAudioPlayerNode  *_player;

    AVAudioPCMBuffer   *_buffer;

    BOOL _sessionInterupted;
    BOOL _sessionConfigPending;
}

- (void)handleInterruption:(NSNotification *)notification;
- (void)handleRouteChange:(NSNotification *)notification;

@end

#pragma mark BMFModsAudioEngine implementation

@implementation BmfLiteDemoAudioEngine

- (instancetype) initWithNSURL:(NSURL *)url VideoFps:(double)fps
{
    NSError* error = nil;
    BOOL success = NO;

    if (self = [super init]) {
        [self initAVAudioSession:fps];

        AVAudioFile *file = [[AVAudioFile alloc] initForReading:url error:&error];
        _buffer = [[AVAudioPCMBuffer alloc] initWithPCMFormat:[file processingFormat] frameCapacity:(AVAudioFrameCount)[file length]];
        success = [file readIntoBuffer:_buffer error:&error];
        NSAssert(success, @"couldn't read audio file, %@", [error localizedDescription]);

        [self createNodeAndInit];

        [self makeEngineConnections];

        NSLog(@"%@", _engine.description);

        __typeof__(self) __weak wself = self;
        [[NSNotificationCenter defaultCenter] addObserverForName:kShouldEnginePauseNotification object:nil queue:[NSOperationQueue mainQueue] usingBlock:^(NSNotification *note) {
            __typeof__(self) __strong sself = wself;
                    
                    if (!sself->_sessionInterupted && sself->_sessionConfigPending) {
                        if (self.playerIsPlaying) return;
                        
                        NSLog(@"Pausing Engine");
                        [sself->_engine pause];
                        [sself->_engine reset];
                        
                        // post notification
                        if ([sself.delegate respondsToSelector:@selector(paused)]) {
                            [sself.delegate paused];
                        }
                    }
        }];
                
        
        [[NSNotificationCenter defaultCenter] addObserverForName:AVAudioEngineConfigurationChangeNotification object:nil queue:[NSOperationQueue mainQueue] usingBlock:^(NSNotification *note) {
            __typeof__(self) __strong sself = wself;
            sself->_sessionConfigPending  = YES;

            if (!sself->_sessionInterupted) {
                NSLog(@"Received a %@ notification!", AVAudioEngineConfigurationChangeNotification);
                [sself makeEngineConnections];
            } else {
                NSLog(@"Session is interrupted, deferring changes");
            }

            if ([self.delegate respondsToSelector:@selector(configureChanged)]) {
                [self.delegate configureChanged];
            }
        }];
    
    }
    return self;
}

- (void)togglePlay
{
    if (![self playerIsPlaying]) {
        [self startEngine];
        [self schedulePlayerContent];
        [_player play];
    } else {
        [_player stop];
        [[NSNotificationCenter defaultCenter] postNotificationName:kShouldEnginePauseNotification object:nil];
    }
}

- (void)stop
{
    [_player stop];
    [[NSNotificationCenter defaultCenter] postNotificationName:kShouldEnginePauseNotification object:nil];
}

- (void)schedulePlayerContent
{
    [_player scheduleBuffer:_buffer atTime:nil options:AVAudioPlayerNodeBufferInterrupts completionHandler:nil];
}

- (void)startEngine
{
    if (!_engine.isRunning) {
        NSError *error;
        BOOL success;
        success = [_engine startAndReturnError:&error];
        NSAssert(success, @"start engine failed, %@", [error localizedDescription]);
    }
}
- (void)handleRouteChange:(NSNotification *)notification
{
}

- (BOOL) playerIsPlaying
{
    return _player.isPlaying;
}

- (void)initAVAudioSession:(double)fps
{
    AVAudioSession *session = [AVAudioSession sharedInstance];
    NSError *error = nil;
    BOOL success = [session setCategory:AVAudioSessionCategoryPlayback error:&error];
    assert(success);
    
    double hwSampleRate = 44100.0;
    success = [session setPreferredSampleRate:hwSampleRate error:&error];
    assert(success);
    NSTimeInterval bufferDuration = 1.0/fps;
    success = [session setPreferredIOBufferDuration:bufferDuration error:&error];
    assert(success);
}

- (void)createNodeAndInit
{
    NSError *error;
    BOOL success = NO;
    _engine = nil;
    _player = nil;

    _player = [[AVAudioPlayerNode alloc] init];
    _engine = [[AVAudioEngine alloc] init];

    [_engine attachNode:_player];
}

- (void)makeEngineConnections
{
    AVAudioMixerNode *mainMixer = [_engine mainMixerNode];
//    AVAudioFormat *stereoFormat = [[AVAudioFormat alloc] initStandardFormatWithSampleRate:44100 channels:2];
    AVAudioFormat *playerFormat = _buffer.format;
    [_engine connect:_player to:mainMixer fromBus:0 toBus:0 format:playerFormat];
}

- (void)handleInterruption:(NSNotification *)notification {
    UInt8 theInterruptionType = [[notification.userInfo valueForKey:AVAudioSessionInterruptionTypeKey] intValue];
    if (theInterruptionType == AVAudioSessionInterruptionTypeBegan) {
        _sessionInterupted = YES;
        [_player stop];
    }
    if (theInterruptionType == AVAudioSessionInterruptionTypeEnded) {
        NSError *error;
        bool success = [[AVAudioSession sharedInstance] setActive:YES error:&error];
        if (success) {
            _sessionInterupted = NO;
            if (_sessionConfigPending) {
                [self makeEngineConnections];
                _sessionConfigPending = NO;
            }
        }
    }
}

@end
