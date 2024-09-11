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

import Foundation
import Combine
import UIKit

func iosModel() -> ModelInfo {
    guard deviceSupportsQuantization else { return ModelInfo.v21Base }
    if deviceHas6GBOrMore { return ModelInfo.xlmbpChunked }
    return ModelInfo.v21Palettized
}

let runningOnMac = ProcessInfo.processInfo.isMacCatalystApp
let deviceHas6GBOrMore = ProcessInfo.processInfo.physicalMemory > 5910000000   // Reported by iOS 17 beta (21A5319a) on iPhone 13 Pro: 5917753344
let deviceHas8GBOrMore = ProcessInfo.processInfo.physicalMemory > 7900000000   // Reported by iOS 17.0.2 on iPhone 15 Pro Max: 8021032960

let deviceSupportsQuantization = {
    if #available(iOS 17, *) {
        true
    } else {
        false
    }
}()


@objc class SDExcutor : NSObject {
    var generation = GenerationContext()

    var downloadProgress : Double = 0
    
    var loader = PipelineLoader(model: iosModel())

    var preparationPhase = "Downloading…"

    var stateSubscriber : Cancellable?

    var completed : Bool = false

    var interval : Double?
    
    var img : CGImage?

    var status : Int = 0

    var result : GenerationResult?

    @objc func generateImage(prompt : String, steps : Double, seed : UInt32) {
        generation.positivePrompt = prompt
        generation.steps = steps
        generation.seed = seed
        
        if case .running = generation.state { return }
        Task {
            generation.state = .running(nil)
            do {
                result = try await generation.generate()
                generation.state = .complete(generation.positivePrompt, result?.image, result?.lastSeed ?? 25, result?.interval)
                img = result?.image
                interval = result?.interval
                completed = true
                status = 3
            } catch {
                generation.state = .failed(error)
                status = 6
            }
        }
    }
    
    @objc func hasCompleted() -> Bool {
        return completed
    }

    @objc func getResult() ->CGImage {
            completed = false
            
            return img!
    }

    @objc func getProcessTime() ->Double {
            return interval!
    }

    @objc func getProgressValue() -> Double {
        return downloadProgress;
    }

    @objc func getPreparationPhase() -> String {
        return preparationPhase;
    }

    @objc func getStatus() -> Int {
        return status
    }

    @objc func loadAndInit() {
        Task.init {
            stateSubscriber = loader.statePublisher.sink { state in
                DispatchQueue.main.async {
                    switch state {
                    case .downloading(let progress):
                        self.preparationPhase = "Downloading model"
                        self.downloadProgress = progress
                        print("load model:%f", self.downloadProgress)
                        self.status = 0
                    case .uncompressing:
                        self.preparationPhase = "Uncompressing model"
                        self.downloadProgress = 1
                        self.status = 1
                    case .readyOnDisk:
                        self.preparationPhase = "Loading model"
                        self.downloadProgress = 1
                        self.status = 2
                    default:
                        self.preparationPhase = "Loading model completed!"
                        self.downloadProgress = 1
                        self.status = 3
                        break
                    }
                }
            }
            do {
                generation.pipeline = try await loader.prepare()
                self.status = 3
            } catch {
                print("load pipeline failed.")
                self.status = 5
            }
        }
    }
}


