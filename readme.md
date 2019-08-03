# RetinaNet in Pytorch

An unofficial implementation of RetinaNet in pytorch. 
Focal Loss for Dense Object Detection.

https://arxiv.org/abs/1708.02002

This repo achieves **35.5%** mAP at nearly 800px resolution with a Resnet-50 backbone. 

![](images/pred_demo.bmp)



## 1. VOC

First, configure *train.json* file, add your root. 

```json
{
    "root_train": "/home1/xyt/dataset/VOC0712_trainval/JPEGImages",
    "root_eval": "/home1/xyt/dataset/VOCdevkit/VOC2007/JPEGImages",
    "list_train": "data/voc_trainval.txt",
    "list_eval": "data/voc_test.txt",
    "name_file": "data/voc_name.txt",

    "load": false,
    "save": true,
    "pretrain": true,
    "freeze_bn": true,
    "epoch_num": [20,10],

    "nbatch_train": 21,
    "nbatch_eval": 21,
    "device": [1,2,3],
    "num_workers": 7,
    
    "lr": 0.01,
    "lr_decay": 0.1,
    "momentum": 0.9,
    "weight_decay": 0.0001,

    "boxarea_th": 16,
    "grad_clip": 5,
    
    "augmentation": true,
    "img_scale_min": 0.6,
    "crop_scale_min": 0.4,
    "aspect_ratio": [0.750, 1.333],
    "remain_min": 0.8
}
```

Then, configure some parameters in *detector.py* file.

```python
# TODO: choose backbone
from backbone import resnet50 as backbone
# TODO: configure Detector
self.train_size = 641
self.eval_size = 641
self.classes = 20   # TODO: total 20 classes exclude background
```

In my experiment, only 30 epochs were performed. Better results can be achieved if it takes longer.
run train to get results. It takes about 5 hours with 3x Titan-Xp. 
run analyze and got mAP@.5: **79.5%**

```python
map_mean
[0.0247 0.1067 0.1873 0.2904 0.3505 0.3594 0.3623 0.3557 0.4027 0.426
 0.4014 0.431  0.3951 0.4495 0.4337 0.442  0.4371 0.4377 0.4598 0.4468
 0.5208 0.5285 0.5298 0.534  0.5378 0.5334 0.5371 0.5396 0.5384 0.5399]
map_50
[0.0569 0.2312 0.359  0.5032 0.6045 0.6036 0.599  0.5976 0.6605 0.6877
 0.6452 0.695  0.6456 0.7122 0.6914 0.6967 0.6851 0.6949 0.7112 0.7065
 0.7782 0.7845 0.7859 0.7894 0.7932 0.7899 0.7919 0.7934 0.7947 0.796 ]
map_75
[0.0187 0.0875 0.1804 0.3028 0.3625 0.3721 0.3789 0.3687 0.426  0.4582
 0.4274 0.4619 0.4205 0.4879 0.4653 0.4788 0.4742 0.4709 0.5015 0.4896
 0.5735 0.5829 0.5853 0.5913 0.5954 0.588  0.5935 0.5975 0.5954 0.5953]
```



## 2. COCO (standard)

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
    "epoch_num": [9,3,2],

    "nbatch_train": 16,
    "nbatch_eval": 16,
    "device": [1,2,3,5,6,7,8,9],
    "num_workers": 16,
    
    "lr": 0.01,
    "lr_decay": 0.1,
    "momentum": 0.9,
    "weight_decay": 0.0001,

    "boxarea_th": 16,
    "grad_clip": 5,
    
    "augmentation": false,
    "img_scale_min": 0.6,
    "crop_scale_min": 0.4,
    "aspect_ratio": [0.750, 1.333],
    "remain_min": 0.8
}
```

Then, configure some parameters in *detector.py* file.

```python
self.classes = 80   # TODO: total 80 classes exclude background
self.train_size = 1025
self.eval_size = 1025
```

It takes about 21 hours with 8x Titan-Xp.  Run analyze to get mAP curves.

```python
map_mean
[0.1248 0.1764 0.1948 0.2136 0.2331 0.2456 0.254  0.2457 0.2561 0.3139
 0.3185 0.3196 0.3225 0.3224]
map_50
[0.2168 0.2968 0.319  0.3452 0.3729 0.3891 0.4021 0.3941 0.4125 0.4815
 0.4863 0.4879 0.4922 0.4924]
map_75
[0.1277 0.1832 0.2077 0.2286 0.2465 0.2631 0.2734 0.2608 0.272  0.3372
 0.3419 0.3438 0.3457 0.3448]
```

Run cocoeval and got mAP: **32.5%**

```python
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.325
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.497
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.349
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.154
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.361
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.438
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.274
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.425
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.453
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.247
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.500
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.581
```



## 3. COCO (data augmentation, longer time)

Like 2 in *train.json* modify key

```json
"epoch_num": [16,6,4],
"augmentation": true,
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





