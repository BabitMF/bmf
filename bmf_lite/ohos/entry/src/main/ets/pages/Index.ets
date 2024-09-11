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
import router from '@ohos.router'
import prompt from '@ohos.promptAction'

@Entry
@Component
struct Index {

  build() {
    Column() {
      Text("Try out BMF Lite!")
        .fontSize($r('app.float.size_20'))
        .width('70%')
        .textAlign(TextAlign.Center)
        .margin({ top: $r('app.float.size_80'), bottom: $r('app.float.size_15') })
        .alignSelf(ItemAlign.Center)
        .fontColor(Color.Black)

      Button($r('app.string.button_player_sr'))
        .id('player_sr')
        .fontSize($r('app.float.size_20'))
        .width('70%')
        .type(ButtonType.Normal)
        .borderRadius($r('app.float.size_5'))
        .margin({ top: $r('app.float.size_15'), bottom: $r('app.float.size_15') })
        .backgroundColor($r('app.color.primary'))
        .fontColor(Color.White)
        .onClick(() => {
          router.pushUrl({ url: 'pages/PlayerPage' })
        })

      Button($r('app.string.button_camera_denoise'))
        .id('camera_denoise')
        .fontSize($r('app.float.size_20'))
        .width('70%')
        .type(ButtonType.Normal)
        .borderRadius($r('app.float.size_5'))
        .margin({ top: $r('app.float.size_15'), bottom: $r('app.float.size_15') })
        .backgroundColor($r('app.color.primary'))
        .fontColor(Color.White)
        .onClick(() => {
          router.pushUrl({ url: 'pages/CameraPage' })
        })

      // Button($r('app.string.button_text_to_pic'))
      //   .fontSize($r('app.float.size_20'))
      //   .width('70%')
      //   .type(ButtonType.Normal)
      //   .borderRadius($r('app.float.size_5'))
      //   .margin({ top: $r('app.float.size_15'), bottom: $r('app.float.size_15') })
      //   .backgroundColor($r('app.color.primary'))
      //   .fontColor(Color.White)
      //   .onClick(() => {
      //     prompt.showToast({ message: $r("app.string.desc_constructing"), duration: 1000 })
      //   })

    }
    .width('100%')
    .alignItems(HorizontalAlign.Center)
    .alignSelf(ItemAlign.Center)
  }

}