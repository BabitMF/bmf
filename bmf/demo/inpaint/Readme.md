# About This Demo
The inpaint demo shows how to quickly integrate a PyTorch model into BMF. The inpaint model we used is called [MAT](https://github.com/fenglinglwb/mat).

# Get Started

### Environment Setup
*   install BMF
*   install dependencies: ffmpeg
*   make sure the GPU environment is ready


### pip install BMF packages

Before installing BMF, please make sure that you have installed Python and pip. It is recommended to use Python 3.8 or newer versions for better compatibility.

To install a GPU supported version of BMF:

```Bash
pip3 install -i https://test.pypi.org/simple/ BabitMF-GPU
%env LIBRARY_PATH=$LIBRARY_PATH:/usr/local/lib/python3.10/dist-packages/bmf/lib
```

### install the FFmpeg libraries

Part of feature in BMF framework utilizes the FFmpeg demuxer/muxer/codec and filter as the built-in modules for video processing, especially, BMF utilizes GPU's hardware codec through ffmpeg. **The capability of ffmpeg is needed in this demo,it's neccessary for users to install supported FFmpeg libraries before using BMF.**

On Ubuntu, we can install ffmpeg through apt.
```Bash
sudo apt update
sudo apt install ffmpeg libdw1
```
If you are using Rocky Linux, please refer to [this article](https://citizix.com/how-to-install-ffmpeg-on-rocky-linux-alma-linux-8/) on how to install ffmpeg.

Make sure the ffmpeg version is >= 4.2

```Bash
ffmpeg -version
```

### Clone the MAT repo

We need the MAT pytorch code and trained model to run the demo. Go to the demo directory and clone the MAT repo.
```bash
cd bmf/demo/inpaint
git clone https://github.com/fenglinglwb/MAT.git 
```
Download the trained model [CelebA-HQ_512.pkl](https://mycuhk-my.sharepoint.com/personal/1155137927_link_cuhk_edu_hk/_layouts/15/onedrive.aspx?ga=1&id=%2Fpersonal%2F1155137927%5Flink%5Fcuhk%5Fedu%5Fhk%2FDocuments%2FRelease%2FMAT%2Fmodels%2FCelebA%2DHQ%5F512%2Epkl&parent=%2Fpersonal%2F1155137927%5Flink%5Fcuhk%5Fedu%5Fhk%2FDocuments%2FRelease%2FMAT%2Fmodels) and put the model in the MAT folder. If you put the model elsewhere, change the `network_pkl` variable to the corresponding path.

### Run the demo
Run the demo use the `test_inpaint.py` script
```python
python test_inpaint.py
```
The demo will rotate the input image clockwise and inpaint the black background generated from the rotation.