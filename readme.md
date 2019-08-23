# RetinaNet in Pytorch

An unofficial implementation of RetinaNet in pytorch. 
Focal Loss for Dense Object Detection.

https://arxiv.org/abs/1708.02002

This repo achieves **35.5%** mAP at nearly 700px resolution with a Resnet-50 backbone. 

![](images/pred_demo.bmp)



## 1. VOC

First, configure *train.json* file, add your root. 

```json
{
    "root_train": "/home/xyt/dataset/VOC0712_trainval/JPEGImages",
    "root_eval": "/home/xyt/dataset/VOCdevkit/VOC2007/JPEGImages",
    "list_train": "data/voc_trainval.txt",
    "list_eval": "data/voc_test.txt",
    "name_file": "data/voc_name.txt",

    "load": false,
    "save": true,
    "pretrain": true,
    "freeze_bn": true,
    "freeze_stages": true,
    "epoches": 30,

    "nbatch_train": 28,
    "nbatch_eval": 28,
    "device": [0,1,2,3],
    "num_workers": 14,

    "lr_base": 0.01,
    "lr_gamma": 0.1,
    "lr_schedule": [16000, 22000],
    "momentum": 0.9,
    "weight_decay": 0.0001,

    "boxarea_th": 32,
    "grad_clip": 3,

    "img_scale_min": 0.6,
    "augmentation": false
}
```

Then, configure some parameters in *detector.py* file.

```python
# TODO: choose backbone
from backbone import resnet50 as backbone
# TODO: configure Detector
self.view_size = 641
self.classes = 20   # TODO: total 20 classes exclude background
```

In my experiment, only 30 epochs were performed. Better results can be achieved if it takes longer.
run train to get results. It takes about 5 hours with 4x Titan-Xp. 
run analyze and got mAP@.5: **78.3%**

```python
map_mean
[0.0279 0.1072 0.2366 0.2952 0.3205 0.3607 0.3728 0.397  0.4291 0.4273
 0.4397 0.4345 0.4677 0.479  0.4766 0.4698 0.4806 0.4749 0.495  0.4937
 0.5388 0.544  0.5518 0.5504 0.5492 0.5497 0.5511 0.5526 0.5526 0.5533]
map_50
[0.0607 0.2096 0.4047 0.5043 0.5375 0.5816 0.5873 0.6389 0.6677 0.666
 0.6751 0.6726 0.7056 0.7249 0.7125 0.7088 0.7181 0.7091 0.7333 0.7277
 0.7715 0.7772 0.7831 0.7804 0.7787 0.7799 0.7824 0.783  0.7834 0.783 ]
map_75
[0.0226 0.0976 0.243  0.2976 0.3285 0.3773 0.3893 0.4223 0.4542 0.4495
 0.472  0.4589 0.4903 0.5113 0.5105 0.4983 0.5154 0.5072 0.5351 0.5297
 0.5761 0.5842 0.5961 0.593  0.5936 0.5936 0.5919 0.5942 0.5938 0.5942]
```



## 2. COCO (1x)

First, configure train.json file, add your root. 

```json
{
    "root_train": "/home1/xyt/dataset/coco17/images",
    "root_eval": "/home1/xyt/dataset/coco17/images",
    "list_train": "data/coco_train2017.txt",
    "list_eval": "data/coco_val2017.txt",
    "name_file": "data/coco_name.txt",

    "load": false,
    "save": true,
    "pretrain": true,
    "freeze_bn": true,
    "freeze_stages": true,
    "epoches": 12,

    "nbatch_train": 16,
    "nbatch_eval": 16,
    "device": [1,2,3,5,6,7,8,9],
    "num_workers": 16,

    "lr_base": 0.01,
    "lr_gamma": 0.1,
    "lr_schedule": [60000, 80000],
    "momentum": 0.9,
    "weight_decay": 0.0001,

    "boxarea_th": 32,
    "grad_clip": 3,

    "img_scale_min": 0.8,
    "augmentation": false
}
```

Then, configure some parameters in *detector.py* file.

```python
self.view_size = 1025
self.classes = 80   # TODO: total 80 classes exclude background
```

It takes about 21 hours with 8x Titan-Xp.  Run analyze to get mAP curves.

```python
map_mean
[0.1094 0.1786 0.1961 0.2203 0.2306 0.2524 0.2529 0.2624 0.3138 0.3196
 0.3273 0.3281]
map_50
[0.1859 0.2873 0.3091 0.3417 0.362  0.3901 0.3866 0.4077 0.4682 0.4769
 0.4858 0.4868]
map_75
[0.1115 0.1865 0.2078 0.234  0.243  0.2671 0.2683 0.2775 0.3347 0.338
 0.3494 0.3494]
```

Run cocoeval and got mAP: **33.1%**

```python
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.331
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.491
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.352
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.174
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.361
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.437
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.277
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.431
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.459
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.271
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.499
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.580
```



## 3. COCO (2x)

Like 2 in *train.json* modify key

```json
"epoches": 24,
"lr_schedule": [120000, 160000],
```

Run train to get results. It takes about 40 hours with 8x Titan-Xp. Run analyze to get mAP curves.

```python
map_mean
[0.1006 0.1625 0.1857 0.2021 0.2178 0.2302 0.2489 0.2487 0.2469 0.2519
 0.2549 0.2636 0.2586 0.2681 0.2678 0.268  0.3308 0.3358 0.3385 0.3412
 0.341  0.344  0.35   0.3507 0.3504 0.3521]
map_50
[0.1855 0.2752 0.3094 0.3398 0.3644 0.3784 0.3974 0.3979 0.3987 0.4052
 0.4131 0.4206 0.418  0.4274 0.4278 0.4246 0.5065 0.5127 0.5147 0.5188
 0.5202 0.5236 0.5302 0.5311 0.5298 0.5308]
map_75
[0.1001 0.1715 0.1944 0.2131 0.2285 0.243  0.2647 0.2641 0.2649 0.2688
 0.269  0.2828 0.2767 0.2882 0.2911 0.2868 0.3544 0.3626 0.3671 0.3686
 0.3697 0.372  0.3783 0.3783 0.3804 0.3813]
```

Run cocoeval and got mAP: **35.5%**

```python
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.355
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.536
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.382
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.194
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.391
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.472
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.294
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.454
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.483
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.291
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.529
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.612
```





