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
import XComponentContext from "../interface/XComponentContext"
import Logger from '../utils/Logger';
import common from '@ohos.app.ability.common';
import abilityAccessCtrl from '@ohos.abilityAccessCtrl';
import { GlobalContext } from '../utils/GlobalContext';
import resourceManager from '@ohos.resourceManager';

@Component
export struct KComponentView {
  @Link xComponentContext: XComponentContext | undefined;
  mXComponentController: XComponentController = new XComponentController()
  @State permissionCamera: Boolean = false;
  @Prop xComponentWidth: number = 384;
  @Prop xComponentHeight: number = 450;
  @Prop usage: number = 0; // 0 player, 1 camera
  private appContext: common.Context = getContext(this);
  atManager = abilityAccessCtrl.createAtManager();

  private tag: string = 'KComponentView';

  private mgr: resourceManager.ResourceManager = {} as resourceManager.ResourceManager;
  private fileDescriptor: resourceManager.RawFileDescriptor | null = null;

  private isCameraCreated: Boolean = false;
  private isCameraRunning: Boolean = false;
  private isXComponentLoaded: Boolean = false;
  private isPlayerRunning: Boolean = false;
  private TAG: string = 'XComponent';

  async aboutToAppear() {
    if (this.usage === 1) {
      if (!this.isCameraCreated) {
        await this.requestPermissionsFn();
      }
    }
  }

  async aboutToDisappear() {
    if (this.usage === 0) {
      if (this.xComponentContext && this.isPlayerRunning) {
        this.xComponentContext.stopPlayer();
        this.xComponentContext.releasePlayer();
      }
    } else if (this.usage === 1) {
      if (this.xComponentContext && this.isCameraRunning) {
        this.xComponentContext.stopCamera();
        this.xComponentContext.releaseCamera();
      }
    }
  }

  async requestPermissionsFn() {
    Logger.info(this.TAG, 'requestPermissionsFn entry');
    if (this.isCameraCreated || this.permissionCamera) return;
    try {
      this.atManager.requestPermissionsFromUser(this.appContext, [
        'ohos.permission.CAMERA',
        'ohos.permission.MICROPHONE'
      ]).then(() => {
        Logger.info(this.TAG, 'request Permissions success!');
        if (this.xComponentContext && !this.isCameraCreated && this.isXComponentLoaded) {
          this.xComponentContext.createCamera();
          this.xComponentContext.startCamera();
          this.isCameraCreated = true;
          this.isCameraRunning = true;
        }
        this.permissionCamera = true;
      })
    } catch (err) {
      Logger.info(this.TAG, `requestPermissionsFromUser call Failed! error: ${err.code}`);
      this.permissionCamera = false;
    }
  }

  build() {
    Column() {
      XComponent({
        id: 'xcomponentId',
        type: XComponentType.SURFACE,
        libraryname: 'nativerender',
        controller: this.mXComponentController
      })
        .onLoad((xComponentContext) => {
          Logger.info(this.TAG, 'XComponent onLoad');
          this.xComponentContext = xComponentContext as XComponentContext;
          this.isXComponentLoaded = true;
          this.mXComponentController.setXComponentSurfaceSize(
            { surfaceWidth: this.xComponentWidth,
              surfaceHeight: this.xComponentHeight }
          );
          if (this.usage === 0) {
            this.xComponentContext.createPlayer();
            this.play();
            this.isPlayerRunning = true;
          } else {
            if (this.permissionCamera && !this.isCameraCreated) {
              this.xComponentContext.createCamera();
              this.xComponentContext.startCamera();
              this.isCameraCreated = true;
              this.isCameraRunning = true;
            }
          }
        })
        .onDestroy(() => {
          console.log('onDestroy');
        })
        .id("xcomponent")
        .width(this.xComponentWidth)
        .height(this.xComponentHeight)
    }
  }

  async play() {
    if (this.xComponentContext) {
      this.mgr = (GlobalContext.getContext().getObject('context') as (common.UIAbilityContext)).resourceManager;
      this.fileDescriptor = await this.mgr.getRawFd('test.mp4');
      this.xComponentContext.setFdSource(this.fileDescriptor.fd,
        this.fileDescriptor.offset, this.fileDescriptor.length);
      this.xComponentContext.startPlayer();
    }
  }

}
