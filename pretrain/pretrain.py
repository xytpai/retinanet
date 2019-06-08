import sys
import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder 
sys.path.append('../')
from extractor import Extractor



# ===============
# TODO: 确定各参数
# ===============
load = False  # 是否使用之前的参数
save = True   # 是否储存新参数
epoch_num = [150000, 300000, 550000] # 迭代步数区间
# 150k-batch:lr=0.1; 150k-batch:lr=0.01; 250k-batch:lr=0.001
step_save = 200    # 多少迭代步后储存
step_eval = 50     # 多少迭代步后评估并记录
lr = 0.1           # 初始学习率
lr_decay = 0.1     # 每一个epoch后权重衰减比例
nbatch_train = 128 # 训练batch大小
nbatch_eval  = 128 # 评估batch大小，由于评估与训练占显存量不一样
size = 224         # 使用多少大小的图像输入
device = [8,9]     # 定义(多)GPU编号列表, 第一个为主设备
root_train = 'D:\\dataset\\ILSVRC2012\\train'
root_eval  = 'D:\\dataset\\ILSVRC2012\\val'
# ===============



# 定义数据集增强或转换步骤
transform_train = transforms.Compose([
    transforms.Resize(480),
    transforms.RandomHorizontalFlip(),
    transforms.RandomResizedCrop(size, scale=(0.53, 1.0)),
    transforms.ColorJitter(brightness=0.03,contrast=0.03,saturation=0.03,hue=0.03),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
transform_eval = transforms.Compose([
    transforms.Resize(size),
    transforms.CenterCrop((size, size)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])



# 定义数据集路径
dataset_train = ImageFolder(root_train, transform=transform_train)
dataset_eval = ImageFolder(root_eval, transform=transform_eval)
loader_train = torch.utils.data.DataLoader(dataset_train, 
    batch_size=nbatch_train, shuffle=True, num_workers=0)
loader_eval = torch.utils.data.DataLoader(dataset_eval, 
    batch_size=nbatch_eval, shuffle=True, num_workers=0) # 需要打乱



# 准备网络
net = Extractor()
net = nn.DataParallel(net, device_ids=device) # 多GPU
net = net.cuda(device[0]) # 先将网络参数传到主设备
log_train_acc = [] # train准确率记录
log_eval_acc = []  # eval准确率记录
if load:
    device_out = 'cuda:%d' % (device[0])
    net.load_state_dict(torch.load('net_e_pretrain.pkl', map_location=device_out))
    log_train_acc = list(np.load('log_train_acc.npy'))
    log_eval_acc  = list(np.load('log_eval_acc.npy'))



# 主循环
step_id = 0 # 记录目前的步数
break_flag = False
criterion = nn.CrossEntropyLoss()
for epoch_id in range(len(epoch_num)):
    while True:
        # Train
        net.train()
        optimizer = torch.optim.SGD(net.parameters(), lr=lr, 
            weight_decay=0.0001, momentum=0.9)
        for i, (inputs, labels) in enumerate(loader_train):
            # Forward and Backward and Optimize
            # 在 DataParallel 下输入会倍自动送入GPU
            labels = labels.cuda(device[0]) # 标签只能进主设备
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            Y_pred = torch.max(outputs,1)[1].cpu()
            labels = labels.cpu()
            acc = float(sum(np.array(Y_pred==labels)))/len(labels)
            print('step:%d,loss:%f,acc:%f' % (step_id, loss, acc))
            loss.backward()
            optimizer.step()
            step_id += 1 
            # Eval
            if (step_id%step_eval == (step_eval-1)):
                log_train_acc.append(float(acc))
                net.eval()
                for i, (inputs, labels) in enumerate(loader_eval):
                    labels = labels.cuda(device[0])
                    outputs = net(inputs)
                    Y_pred = torch.max(outputs,1)[1].cpu()
                    labels = labels.cpu()
                    acc = float(sum(np.array(Y_pred==labels)))/len(labels)
                    log_eval_acc.append(float(acc))
                    net.train()
                    # 随机采一次后直接跳出
                    # 因此尽量扩大测试batch大小
                    break
            # Save
            if (step_id%step_save == (step_save-1)) and save:
                torch.save(net.state_dict(),'net_e_pretrain.pkl')
                if len(log_train_acc)>0:
                    np.save('log_train_acc.npy', log_train_acc)
                    np.save('log_eval_acc.npy', log_eval_acc)
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