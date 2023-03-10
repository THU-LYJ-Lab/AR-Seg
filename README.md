# AR-Seg

> Efficient Semantic Segmentation by Altering Resolutions for Compressed Videos <br>
[Yubin Hu](https://github.com/AlbertHuyb), [Yuze He](https://github.com/hyz317/), Yanghao Li, Jisheng Li, Yuxing Han, Jiangtao Wen, Yong-Jin Liu <br>
CVPR 2023

## Introduction 

AR-Seg is an efficient video semantic segmentation framework for compressed videos. It consists of an HR branch for keyframes and an LR branch for non-keyframes.
![](./static/diagram.png)
We design a Cross Resolution Feature Fusion (CReFF) module and a Feature Similarity Training (FST) strategy to compensate for the performance drop because of low-resolution. 
![](./static/method.png)

## Dataset & Pre-processing

Please refer to the [documentation](./pre-process/README.md).

