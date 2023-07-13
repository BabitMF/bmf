/** \page GPU Module GPU处理模块

BMF provides several common filtering modules accelerated by GPU:
- Flip
- flip
- rotate
- crop
- blur

These modules are located in bmf/example/gpu_module/. The demo code showing how to use these modules is in `test_gpu_module.py`. You can also try them on Colab [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/eefengwei/colab_tutorials/blob/main/colab_tutorial_cd.ipynb). The modules accept nv12/yuv420p/rgb formats with 8/10 bits per component.

## Scale

The scale module can resize an image.

**Scale Options**

***size***: Target size of the image, can be of format `WxH`, `W*H` or `W,H`

***algo***(optional): Interpretation algorithm, can be one of `area`, `cubic`, `linear` and `nearest`. If unspecified, `linear` will be used by default.

## Flip

Flip the image in the specified direction.

**Flip Options**

***direction***: The direction of the flip. Supports `h`, `horizontal`, `v`, `vertical` or `both` (flip both vertically and horizontally). E.g. `.module('flip_gpu', {'direction': 'v'})` flips the image vertically

## Rotate

Rotate the image in a certain angle. The angle can be represented as degrees or radians. By default, the image will be rotated around the image center. Note that the empty area will be fill with 0. Rotation will not change the aspect ratio of the image, i.e. if you rotate a 1920x1080 image by 90 degrees, the output image is still 1920x1080 rather than 1080x1920

**Rotate Options**

***angle_deg***: The angle of the rotation. A positive angle value will rotate the image clockwise, a negative angle value anti-clockwise. The angle can be a floating point value.

***angle***: The radius of the rotation. e.g. setting `{'angle': 'pi/8'}` will effectively rotate the image by 45 degrees clockwise.

***center***: The point around which the image is rotated. By default, the center is the image center `w/2,h/2`, where `w` and `h` are the width and height of the image.

***scale***: Scaling factor of the image. Default is `1`, which means do not scale the image. `{'scale': 1.5}` will zoom in the image by 1.5x.

***algo***: Interpolation algorithm. Supports `'cubic'`, `'linear'` and `'nearest'`.

## Crop

Crops the input image into the given size. Example: `module('crop_gpu', {'x': 960, 'y': 540, 'width': 640, 'height': 480})`

**Crop Options**

***x, y***: The coordinate of the cropping area's upper-left corner.

***width, height***: Width and height of the cropping area.

## Blur

Blur the input image using one of the supported algorithm (gaussian, average and median). Gaussian blur example: `module('blur_gpu', {'op': 'gblur', 'sigma': [0.7, 0.7], 'size': [5, 5]})`

**Blur Options**

***op***: The blur algorithm to be used. Supports `'gblur'` (gaussian blur), `'avgblur'` (average blur) and `'median'` (median blur).

***size***: The width and height of the blur kernel. Should be of format `[W, H]`. Default size is `[1, 1]` This option applies to all blur algorithms.

***planes***: Specifies which image plane should be blurred. The value should be a bit mask, e.g. if you want to blur all three planes of a yuv420p image, you should set `'planes': 0x7` or `'planes': 0b111`. Default value is `0xf`. This option applies to all blur algorithms.

***sigma***: Gaussian kernel standard deviation. Should be of format `[X, Y]`, float data type. This option only applies to the `gblur` op.

***anchor***: Kernel anchor. Indicates the relative position of a filtered point within the kernel. Should be of format `[X, Y]`. Default is `[-1, -1]` which indicates kernel center. This option only applies to `avgblur` op.

***radius***: Alias for the ***size*** option. Radius only applies to `median` op.