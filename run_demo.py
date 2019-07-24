import torch
import numpy as np 
import os
import json
from PIL import Image
import torchvision.transforms as transforms
from utils_box.dataset import show_bbox, corner_fix
from detector import Detector


with open('train.json', 'r') as load_f:
    cfg = json.load(load_f)
net = Detector(pretrained=False)



# TODO: ============================
# net.nms_th = 0.05
# ==================================



net.load_state_dict(torch.load('net.pkl', map_location='cpu'))
net.eval()


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))])


with open(cfg['name_file']) as f:
    lines = f.readlines()
LABEL_NAMES = []
for line in lines:
    LABEL_NAMES.append(line.strip())


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
        img = img.view(1, img.shape[0], img.shape[1], img.shape[2])
        with torch.no_grad():
            cls_i_preds, cls_p_preds, reg_preds = net(img)
            name = 'images/pred_'+filename.split('.')[0]+'.bmp'
            reg_preds[0] /= scale
            show_bbox(img_cpy, reg_preds[0].cpu(), cls_i_preds[0].cpu(), LABEL_NAMES, name)

