### bmf- video_quality_assessment demo
!!!!!!!! Before running, you need to create models dir and download the onnx file to the created dir. The onnx file path is releases/download/files/vqa_4kpgc_1.onnx

1. Algorithm introduction:
The vqa model evaluates video quality for 4kpgc scenes, and carries out targeted algorithm design and optimization for specific spatiotemporal distortion characteristics.The model is trained and tested based on NAIC(National Artifical Intelligence Challenge) competition data. The average value of the model on srcc, plcc and ur(usability ratio) reaches 0.855, winning the championship of the AI+ Video Quality evaluation circuit of the National Artificial Intelligence Competition. Details: https://naic.pcl.ac.cn/contest/17/53

2. Get Started quickly:
By running vqa_demo.py file, you can quickly complete the experience of video quality assessment. The quality score ranges from 0 to 10. The higher the score, the lower the distortion degree and the better the video quality.

4. Requirements
    os
    bmf
    sys
    time
    json
    numpy
    onnxruntime
