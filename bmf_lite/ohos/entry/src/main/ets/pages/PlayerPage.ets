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
import { GlobalContext } from '../utils/GlobalContext'
import media from '@ohos.multimedia.media';
import display from '@ohos.display';
import XComponentContext from "../interface/XComponentContext"
import {KComponentView} from "../component/KComponentView"
import Logger from '../utils/Logger';
// import apis from 'libnativerender.so'


@Entry
@Component
export struct PlayerPage {
  private screenHeight: number = 0;
  private screenWidth: number = 0;
  @State xComponentWidth: number = 384;
  @State xComponentHeight: number = 450;
  @State xComponentContext: XComponentContext | undefined = undefined;

  @State private algRunning: boolean = false;

  aboutToAppear() {
    let mDisplay = display.getDefaultDisplaySync();
    this.screenWidth = px2vp(mDisplay.width);
    this.screenHeight = px2vp(mDisplay.height);
    this.initXComponentSize();
  }

  initXComponentSize(): void {
    this.xComponentWidth = this.screenWidth;
    this.xComponentHeight = this.screenWidth * 1280 / 720;
  }

  aboutToDisappear() {
  }

  onPageHide() {
  }

  onPageShow() {
  }

  build() {
    Column() {
      Stack() {
        KComponentView({
          xComponentWidth: this.xComponentWidth,
          xComponentHeight: this.xComponentHeight,
          xComponentContext: this.xComponentContext,
          usage: 0
        })
      }.backgroundColor(Color.Black)
      .width(this.xComponentWidth)
      .height(this.xComponentHeight)

      Row() {
        Text($r('app.string.hint_unprocessed'))
          .width($r('app.float.size_50'))
          .height($r('app.float.size_24'))
        Text($r('app.string.hint_processed'))
          .width($r('app.float.size_50'))
          .height($r('app.float.size_24'))
      }
      .width('100%')
      .height('3%')
      .justifyContent(FlexAlign.SpaceEvenly)
      .visibility(this.algRunning ? Visibility.Visible : Visibility.Hidden)

      Row() {
        Button($r('app.string.button_start_sr'))
          .width($r('app.float.size_80'))
          .height($r('app.float.size_48'))
          .onClick(() => {
            if (!this.algRunning && this.xComponentContext) {
              this.xComponentContext.startAlgorithm(GlobalContext.ALG_SUPER_RESOLUTION);
              this.algRunning = true;
            }
          })
        Button($r('app.string.button_stop_sr'))
          .width($r('app.float.size_80'))
          .height($r('app.float.size_48'))
          .onClick(() => {
            if (this.algRunning && this.xComponentContext) {
              this.xComponentContext.stopAlgorithm(GlobalContext.ALG_SUPER_RESOLUTION);
              this.algRunning = false;
            }
          })
      }
      .width('100%')
      .height('6%')
      .justifyContent(FlexAlign.SpaceEvenly)
    }
    .width('100%')
    .height('100%')
  }

}