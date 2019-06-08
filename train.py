import numpy as np
import torch
import torchvision.transforms as transforms
from dataset import Dataset_Detection
from extractor import Extractor
from detector import Detector
from encoder import Encoder
from loss import focal_loss_detection



# ===============
# TODO: 确定各参数
# ===============
load = False  # 是否使用之前的参数
save = True   # 是否储存新参数
pretrain = True
freeze_bn = False
epoch_num = [60000, 80000, 90000] # 迭代步数区间
step_save = 200    # 多少迭代步后储存
step_eval = 50     # 多少迭代步后评估并记录
lr = 0.1           # 初始学习率
lr_decay = 0.1     # 每一个epoch后权重衰减比例
nbatch_train = 24  # 训练batch大小
nbatch_eval  = 24  # 评估batch大小，由于评估与训练占显存量不一样
size = 513         # 使用多少大小的图像输入
iou_th = (0.3, 0.5)
device = [0]     # 定义(多)GPU编号列表, 第一个为主设备
root_train = 'D:\\dataset\\VOC2012\\JPEGImages'
list_train = 'data/voc_train.txt'
root_eval  = 'D:\\dataset\\VOC2012\\JPEGImages'
list_eval  = 'data/voc_val.txt'
# ===============



# 定义数据集增强或转换步骤
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))])
dataset_train = Dataset_Detection(root, list_train, size=size, train=True, transform=transform)
dataset_eval = Dataset_Detection(root, list_eval, size=size, train=False, transform=transform)
loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=nbatch_train, 
                    shuffle=True, num_workers=0, collate_fn=dataset_train.collate_fn)
loader_eval = torch.utils.data.DataLoader(dataset_eval, batch_size=nbatch_eval, 
                    shuffle=True, num_workers=0, collate_fn=dataset_eval.collate_fn)



# 准备网络
net_e = Extractor()
net_e = nn.DataParallel(net_e, device_ids=device)
net_e = net_e.cuda(device[0])
net_d = Detector()
net_d = nn.DataParallel(net_d, device_ids=device)
net_d = net_d.cuda(device[0])
log_train_loss = []
log_eval_loss = []
if load:
    device_out = 'cuda:%d' % (device[0])
    net_e.load_state_dict(torch.load('net_e.pkl', map_location=device_out))
    net_d.load_state_dict(torch.load('net_d.pkl', map_location=device_out))
    log_train_loss = list(np.load('log_train_loss.npy'))
    log_eval_loss  = list(np.load('log_eval_loss.npy'))
else:
    if pretrain:
        net_e.load_state_dict(torch.load('pretrain/net_e_pretrain.pkl', map_location=device_out))
if freeze_bn:
    net_e.freeze_bn()



# 主循环
step_id = 0 # 记录目前的步数
break_flag = False
encoder = Encoder(train_iou_th=iou_th, train_size=(size, size))
for epoch_id in range(len(epoch_num)):
    while True:
        # Train
        net_e.train()
        net_d.train()
        opt_e = torch.optim.SGD(net_e.parameters(), lr=lr, momentum=0.9, weight_decay=0.0001)
        opt_d = torch.optim.SGD(net_d.parameters(), lr=lr, momentum=0.9, weight_decay=0.0001)
        for i, (img, bbox, label) in enumerate(loader_train):
            # zero grad
            opt_e.zero_grad()
            opt_d.zero_grad()
            # get output and loss
            out = net_e(img, classify=False)
            cls_out, reg_out = net_d(out)
            cls_targets, reg_targets = encoder.encode(label, bbox)
            cls_targets, reg_targets = cls_targets.cuda(device[0]), reg_targets.cuda(device[0])
            loss = focal_loss_detection(cls_out, reg_out, cls_targets, reg_targets)
            # opt
            loss.backward()
            opt_e.step()
            opt_d.step()
            # print
            print('step:%d, loss:%f' % (step_id, loss))
            # step acc
            step_id += 1 
            # Eval
            if (step_id%step_eval == (step_eval-1)):
                log_train_loss.append(float(loss))
                net_e.eval()
                net_d.eval()
                for i, (img, bbox, label) in enumerate(loader_eval):
                    out = net_e(img, classify=False)
                    cls_out, reg_out = net_d(out)
                    cls_targets, reg_targets = encoder.encode(label, bbox)
                    cls_targets, reg_targets = cls_targets.cuda(device[0]), reg_targets.cuda(device[0])
                    loss = focal_loss_detection(cls_out, reg_out, cls_targets, reg_targets)
                    log_eval_loss.append(float(loss))
                    net_e.train()
                    net_d.train()
                    # 随机采一次后直接跳出
                    # 因此尽量扩大测试batch大小
                    break
            # Save
            if (step_id%step_save == (step_save-1)) and save:
                torch.save(net_e.state_dict(),'net_e.pkl')
                torch.save(net_d.state_dict(),'net_d.pkl')
                if len(log_train_loss)>0:
                    np.save('log_train_loss.npy', log_train_loss)
                    np.save('log_eval_loss.npy', log_eval_loss)
            # Break inner
            if step_id >= epoch_num[epoch_id]:
                break_flag = True
                break
        # Break outer
        if break_flag:
            break_flag = False
            break
    # 衰减学习率
    lr *= lr_decay
