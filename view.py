import torch
import numpy as np
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
import cv2, math
from extractor import Extractor
from detector import Detector
from encoder import Encoder
from dataset import show_bbox



# TODO: define parameter
img_path = 'img/0.jpg'



# define network
net_e = Extractor()
net_d  = Detector()
encoder = Encoder()



# load
net_e.load_state_dict(torch.load('net_e.pkl', map_location='cpu'))
net_d.load_state_dict(torch.load('net_d.pkl', map_location='cpu'))
net_e.eval()
net_d.eval()



# prepare img
img = cv2.imread(img_path)
height = img.shape[0]
width = img.shape[1]
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = img.reshape((1, height, width, 3)).transpose((0, 3, 1, 2))  # [1, 3, H, W]



# TODO: get output
x = torch.from_numpy(img).float().div(255)
normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
x[0] = normalize(x[0])
out = net_e(x, classify=False)
cls_out, reg_out = net_d(out)
cls_i_preds, cls_p_preds, reg_preds = encoder.decode(cls_out, reg_out, (height, width))



print(cls_i_preds[0].shape)
print(cls_p_preds[0].shape)
print(reg_preds[0].shape)
VOC_LABEL_NAMES = (
    'background',#0
    'aeroplane',#1
    'bicycle',#2
    'bird',#3
    'boat',#4
    'bottle',#5
    'bus',#6
    'car',#7
    'cat',#8
    'chair',#9
    'cow',#10
    'diningtable',#11
    'dog',#12
    'horse',#13
    'motorbike',#14
    'person',#15
    'pottedplant',#16
    'sheep',#17
    'sofa',#18
    'train',#19
    'tvmonitor'#20
    )
show_bbox(img[0]/255.0, reg_preds[0], cls_i_preds[0], VOC_LABEL_NAMES)
