import numpy as np
import torch
import json
import time
import torchvision.transforms as transforms
from utils_box.dataset import Dataset_CSV
from utils_box.eval_csv import eval_detection
from detector import Detector, get_loss, get_pred


# Read train.json and set current GPU (for nms_cuda)
with open('train.json', 'r') as load_f:
    cfg = json.load(load_f)
torch.cuda.set_device(cfg['device'][0])


# Prepare the network and read log
net = Detector(pretrained=cfg['pretrain'])
log = []
device_out = 'cuda:%d' % (cfg['device'][0])
if cfg['load']:
    net.load_state_dict(torch.load('net.pkl', map_location=device_out))
    log = list(np.load('log.npy'))
net = torch.nn.DataParallel(net, device_ids=cfg['device'])
net = net.cuda(cfg['device'][0])
net.train()


# Get train/eval dataset and dataloader
dataset_train = Dataset_CSV(cfg['root_train'], cfg['list_train'], cfg['name_file'], 
    size=net.module.view_size, train=True, normalize=True, 
    boxarea_th = cfg['boxarea_th'], 
    img_scale_min = cfg['img_scale_min'], 
    crop_scale_min = cfg['crop_scale_min'], 
    aspect_ratio = cfg['aspect_ratio'], 
    remain_min = cfg['remain_min'],
    augmentation = cfg['augmentation'])
dataset_eval = Dataset_CSV(cfg['root_eval'], cfg['list_eval'], cfg['name_file'], 
    size=net.module.view_size, train=False, normalize=True)
loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=cfg['nbatch_train'], 
                    shuffle=True, num_workers=cfg['num_workers'], collate_fn=dataset_train.collate_fn)
loader_eval = torch.utils.data.DataLoader(dataset_eval, batch_size=cfg['nbatch_eval'], 
                    shuffle=False, num_workers=0, collate_fn=dataset_eval.collate_fn)


# Prepare optimizer
lr = cfg['lr']
lr_decay = cfg['lr_decay']
opt = torch.optim.SGD(net.parameters(), lr=lr, 
            momentum=cfg['momentum'], weight_decay=cfg['weight_decay'])


# Run warmup
if not cfg['load']:
    WARM_UP_ITERS = 500
    WARM_UP_FACTOR = 1.0 / 3.0
    if cfg['freeze_bn']:
        net.module.backbone.freeze_bn()
    for i, (img, bbox, label, scale, oh, ow) in enumerate(loader_train):
        alpha = float(i) / WARM_UP_ITERS
        warmup_factor = WARM_UP_FACTOR * (1.0 - alpha) + alpha
        for param_group in opt.param_groups:
            param_group['lr'] = lr * warmup_factor
        time_start = time.time()
        opt.zero_grad()
        temp = net(img, label, bbox)
        loss = get_loss(temp)
        loss.backward()
        clip = cfg['grad_clip']
        torch.nn.utils.clip_grad_norm_(net.parameters(), clip)
        opt.step()
        maxmem = int(torch.cuda.max_memory_allocated(device=cfg['device'][0]) / 1024 / 1024)
        time_end = time.time()
        totaltime = int((time_end - time_start) * 1000)
        print('warmup: step:%d/%d, lr:%f, loss:%f, maxMem:%dMB, time:%dms' % \
                    (i, WARM_UP_ITERS, lr * warmup_factor, loss, maxmem, totaltime))
        if i >= WARM_UP_ITERS:
            break


# Run epoch
epoch = 0
for epoch_num in cfg['epoch_num']: # 3 for example

    for param_group in opt.param_groups:
        param_group['lr'] = lr

    for e in range(epoch_num):

        if cfg['freeze_bn']:
            net.module.backbone.freeze_bn()

        # Train
        for i, (img, bbox, label, scale, oh, ow) in enumerate(loader_train):
            time_start = time.time()
            opt.zero_grad()
            temp = net(img, label, bbox)
            loss = get_loss(temp)
            loss.backward()
            clip = cfg['grad_clip']
            if clip > 0:
                torch.nn.utils.clip_grad_norm_(net.parameters(), clip)
            opt.step()
            maxmem = int(torch.cuda.max_memory_allocated(device=cfg['device'][0]) / 1024 / 1024)
            time_end = time.time()
            totaltime = int((time_end - time_start) * 1000)
            print('epoch:%d, step:%d/%d, loss:%f, maxMem:%dMB, time:%dms' % \
                (epoch, i*cfg['nbatch_train'], len(dataset_train), loss, maxmem, totaltime))

        # Eval
        with torch.no_grad():
            net.eval()
            pred_bboxes = []
            pred_labels = []
            pred_scores = []
            gt_bboxes = []
            gt_labels = []
            for i, (img, bbox, label, scale, oh, ow) in enumerate(loader_eval):
                temp = net(img)
                cls_i_preds, cls_p_preds, reg_preds = get_pred(temp, 
                        net.module.nms_th, net.module.nms_iou, oh, ow)
                for idx in range(len(cls_i_preds)):
                    cls_i_preds[idx] = cls_i_preds[idx].cpu().detach().numpy()
                    cls_p_preds[idx] = cls_p_preds[idx].cpu().detach().numpy()
                    reg_preds[idx] = reg_preds[idx].cpu().detach().numpy()
                _boxes = []
                _label = []
                for idx in range(bbox.shape[0]):
                    mask = label[idx] > 0
                    _boxes.append(bbox[idx][mask].detach().numpy())
                    _label.append(label[idx][mask].detach().numpy())
                pred_bboxes += reg_preds
                pred_labels += cls_i_preds
                pred_scores += cls_p_preds
                gt_bboxes += _boxes
                gt_labels += _label
                print('  Eval: {}/{}'.format(i*cfg['nbatch_eval'], len(dataset_eval)), end='\r')
            ap_iou = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
            ap_res = []
            for iou_th in ap_iou:
                res = eval_detection(pred_bboxes, pred_labels, 
                            pred_scores, gt_bboxes, gt_labels, iou_th=iou_th)
                ap_res.append(res)
            ap_sum = 0.0
            for i in range(len(ap_res)):
                ap_sum += float(ap_res[i]['map'])
            map_mean = ap_sum / float(len(ap_res))
            map_50 = float(ap_res[0]['map'])
            map_75 = float(ap_res[5]['map'])
            print('map_mean:', map_mean, 'map_50:', map_50, 'map_75:', map_75)
            log.append([map_mean, map_50, map_75])
            net.train()
        
        # Save
        if cfg['save']:
            torch.save(net.module.state_dict(),'net.pkl')
            if len(log)>0:
                np.save('log.npy', log)
         
        epoch += 1

    lr *= lr_decay
