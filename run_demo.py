import torch
import numpy as np 
import os
import json
from PIL import Image
import torchvision.transforms as transforms
from utils_box.dataset import show_bbox, corner_fix
from detector import Detector, get_loss, get_pred


# Read train.json and set current GPU (for nms_cuda) and prepare the network
DEVICE = 0 # set device
with open('train.json', 'r') as load_f:
    cfg = json.load(load_f)
torch.cuda.set_device(DEVICE)
net = Detector(pretrained=False)
net.load_state_dict(torch.load('net.pkl', map_location='cpu'))
net = net.cuda()
net.eval()


# TODO: Set nms_th
# net.nms_th = 0.05
# ==================================


# Read LABEL_NAMES
with open(cfg['name_file']) as f:
    lines = f.readlines()
LABEL_NAMES = []
for line in lines:
    LABEL_NAMES.append(line.strip())


# Run
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))])

for filename in os.listdir('images/'):
    if filename.endswith('jpg'):
        img = Image.open(os.path.join('images/', filename))
        if img.mode != 'RGB':
            img = img.convert('RGB')
        _boxes = torch.zeros(0,4)
        img_cpy = img.copy()
        img_cpy = transforms.ToTensor()(img_cpy)
        img, _boxes, scale = corner_fix(img, _boxes, net.eval_size)
        img = transform(img)
        img = img.view(1, img.shape[0], img.shape[1], img.shape[2]).cuda()
        with torch.no_grad():
            temp = net(img)
            cls_i_preds, cls_p_preds, reg_preds = get_pred(temp, 
                net.nms_th, net.nms_iou, net.eval_size)
            name = 'images/pred_'+filename.split('.')[0]+'.bmp'
            reg_preds[0] /= scale
            show_bbox(img_cpy, reg_preds[0].cpu(), cls_i_preds[0].cpu(), LABEL_NAMES, name)

