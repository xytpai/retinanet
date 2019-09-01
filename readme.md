# RetinaNet in Pytorch

An unofficial implementation of RetinaNet in pytorch. 
Focal Loss for Dense Object Detection.

https://arxiv.org/abs/1708.02002

This repo achieves **35.1%** mAP at nearly 700px resolution with a Resnet-50 backbone. 

| paper (600px) | detectron (800px) | ours (nearly 700px) |
| :--: | :---------: | :--: |
| 34 | 35.7 | **35.1** |

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
    "freeze_stages": 1,
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
run analyze and got mAP@.5: **79.1%**

```python
map_mean
[0.0372 0.1881 0.312  0.3404 0.3744 0.3819 0.3969 0.4198 0.4351 0.452
 0.4424 0.4602 0.4708 0.4692 0.4735 0.4725 0.4683 0.4809 0.4755 0.4909
 0.5218 0.5249 0.525  0.5219 0.5261 0.5251 0.524  0.5249 0.5247 0.5258]
map_50
[0.0861 0.3869 0.5774 0.6091 0.6529 0.6574 0.6697 0.6924 0.7126 0.7339
 0.7158 0.7387 0.7425 0.7427 0.7431 0.7493 0.7435 0.7473 0.7394 0.7567
 0.7884 0.7925 0.7889 0.7878 0.792  0.792  0.7868 0.7899 0.7905 0.7909]
map_75
[0.026  0.1617 0.2952 0.3366 0.3847 0.393  0.4169 0.4451 0.4621 0.4809
 0.47   0.4879 0.5032 0.5026 0.5053 0.5108 0.502  0.5129 0.5166 0.5329
 0.5632 0.5661 0.5642 0.5654 0.5667 0.5661 0.5654 0.565  0.5654 0.5656]
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
    "freeze_stages": 1,
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
[0.1401 0.2009 0.2189 0.2455 0.2534 0.2556 0.2672 0.2726 0.3224 0.3291
 0.3329 0.3343]
map_50
[0.2548 0.3444 0.3679 0.4026 0.4157 0.4158 0.4316 0.4363 0.5043 0.5142
 0.5174 0.5193]
map_75
[0.1392 0.2068 0.2273 0.257  0.2668 0.2696 0.2831 0.2906 0.3442 0.3549
 0.3576 0.3588]
```

Run cocoeval and got mAP: **33.7%**

```python
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.337
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.524
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.361
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.178
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.370
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.445
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.277
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.432
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.462
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.275
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.501
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.583
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
[0.1373 0.2022 0.2218 0.2404 0.2539 0.2638 0.26   0.2638 0.2783 0.2777
 0.2791 0.2865 0.285  0.2762 0.2888 0.2982 0.3398 0.3424 0.345  0.3474
 0.3459 0.3489 0.3494 0.3489]
map_50
[0.2521 0.3387 0.3788 0.3994 0.4142 0.426  0.4256 0.4268 0.446  0.4431
 0.4511 0.462  0.4533 0.4433 0.4621 0.4688 0.5253 0.5299 0.5315 0.5354
 0.5335 0.5375 0.538  0.5378]
map_75
[0.1364 0.213  0.2323 0.2508 0.2689 0.2782 0.2765 0.2801 0.2952 0.2973
 0.2955 0.3054 0.2996 0.2947 0.3091 0.3196 0.3636 0.3666 0.3691 0.3726
 0.37   0.3725 0.3739 0.3731]
```

Run cocoeval and got mAP: **35.1%**

```python
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.351
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.541
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.374
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.189
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.385
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.464
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.288
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.445
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.474
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.291
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.519
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.598
```

