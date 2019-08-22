import math
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from utils_box.anchors import gen_anchors, box_iou
from libs.sigmoid_focal_loss import sigmoid_focal_loss
from libs.smooth_l1_loss import smooth_l1_loss
from libs.nms import box_nms 
# TODO: choose backbone
from backbone import resnet50 as backbone



class Detector(nn.Module):
    def __init__(self, pretrained=False):
        super(Detector, self).__init__()

        # ---------------------------
        # TODO: Param
        self.a_hw = [
            [32.00, 32.00],
            [40.32, 40.32],
            [50.80, 50.80],

            [22.63, 45.25],
            [28.51, 57.02],
            [35.92, 71.84],

            [45.25, 22.63],
            [57.02, 28.51],
            [71.84, 35.92],
        ]
        self.scales = 5
        self.first_stride = 8
        self.view_size = 1025
        self.iou_th = (0.4, 0.5)
        self.classes = 80
        self.nms_th = 0.05
        self.nms_iou = 0.5
        self.max_detections = 3000
        self.balanced_fpn = False
        # ---------------------------

        # fpn =======================================================
        self.backbone = backbone(pretrained=pretrained)
        self.relu = nn.ReLU(inplace=True)
        self.conv_out6 = nn.Conv2d(2048, 256, kernel_size=3, padding=1, stride=2)
        self.conv_out7 = nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=2)
        self.prj_5 = nn.Conv2d(2048, 256, kernel_size=1)
        self.prj_4 = nn.Conv2d(1024, 256, kernel_size=1)
        self.prj_3 = nn.Conv2d(512, 256, kernel_size=1)
        self.conv_5 =nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv_4 =nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv_3 =nn.Conv2d(256, 256, kernel_size=3, padding=1)

        # head =======================================================
        self.conv_cls = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, len(self.a_hw)*self.classes, kernel_size=3, padding=1))
        self.conv_reg = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, len(self.a_hw)*4, kernel_size=3, padding=1))

        # reinit head =======================================================
        for layer in self.conv_cls.children():
            if isinstance(layer, nn.Conv2d):
                nn.init.constant_(layer.bias, 0)
                nn.init.normal_(layer.weight, mean=0, std=0.01)
        for layer in self.conv_reg.children():
            if isinstance(layer, nn.Conv2d):
                nn.init.constant_(layer.bias, 0)
                nn.init.normal_(layer.weight, mean=0, std=0.01)
        pi = 0.01
        _bias = -math.log((1.0-pi)/pi)
        nn.init.constant_(self.conv_cls[-1].bias, _bias)

        # generate anchors =======================================================
        self._view_anchors_yxyx, self._view_anchors_yxhw = \
            gen_anchors(self.a_hw, self.scales, self.view_size, self.first_stride)
        self.view_hwan = self._view_anchors_yxyx.shape[0]
        self.register_buffer('view_anchors_yxyx', self._view_anchors_yxyx)
        self.register_buffer('view_anchors_yx', self._view_anchors_yxhw[:, :2])
        self.register_buffer('view_anchors_hw', self._view_anchors_yxhw[:, 2:])


    def upsample(self, input):
        return F.interpolate(input, size=(input.shape[2]*2-1, input.shape[3]*2-1),
                    mode='bilinear', align_corners=True) # input size must be odd
    

    def forward(self, x, loc, label_class=None, label_box=None):
        '''
        Param:
        x:           FloatTensor(batch_num, 3, H, W)
        loc:         FloatTensor(batch_num, 4)
        label_class: LongTensor(batch_num, N_max) or None
        label_box:   FloatTensor(batch_num, N_max, 4) or None

        Return 1:
        loss: FloatTensor(batch_num)

        Return 2:
        cls_i_preds: LongTensor(batch_num, topk)
        cls_p_preds: FloatTensor(batch_num, topk)
        reg_preds:   FloatTensor(batch_num, topk, 4)
        '''
        
        C3, C4, C5 = self.backbone(x)
        
        P5 = self.prj_5(C5)
        P4 = self.prj_4(C4)
        P3 = self.prj_3(C3)
        
        P4 = P4 + self.upsample(P5)
        P3 = P3 + self.upsample(P4)

        P3 = self.conv_3(P3)
        P4 = self.conv_4(P4)
        P5 = self.conv_5(P5)

        P6 = self.conv_out6(C5)
        P7 = self.conv_out7(self.relu(P6))

        if self.balanced_fpn:
            # kernel_size, stride, padding, dilation, False, False
            P3 = F.max_pool2d(P3, 3, 2, 1, 1, False, False)
            P5 = self.upsample(P5)
            P4 = (P3 + P4 + P5) / 3.0
            # kernel_size, stride, padding, False, False
            P5 = F.avg_pool2d(P4, 3, 2, 1, False, False)
            P3 = self.upsample(P4)

        pred_list = [P3, P4, P5, P6, P7]
        assert len(pred_list) == self.scales

        cls_out = []
        reg_out = []
        for item in pred_list:
            cls_i = self.conv_cls(item)
            reg_i = self.conv_reg(item)
            # cls_i: [b, an*classes, H, W] -> [b, H*W*an, classes]
            cls_i = cls_i.permute(0,2,3,1).contiguous()
            cls_i = cls_i.view(cls_i.shape[0], -1, self.classes)
            # reg_i: [b, an*4, H, W] -> [b, H*W*an, 4]
            reg_i = reg_i.permute(0,2,3,1).contiguous()
            reg_i = reg_i.view(reg_i.shape[0], -1, 4)
            cls_out.append(cls_i)
            reg_out.append(reg_i)
        
        # cls_out[b, hwan, classes]
        # reg_out[b, hwan, 4]
        cls_out = torch.cat(cls_out, dim=1)
        reg_out = torch.cat(reg_out, dim=1)
        
        if (label_class is not None) and (label_box is not None):
            targets_cls, targets_reg = self._encode(label_class, label_box, loc) # (b, hwan), (b, hwan, 4)
            mask_cls = targets_cls > -1 # (b, hwan)
            mask_reg = targets_cls > 0 # (b, hwan)
            num_pos = torch.sum(mask_reg, dim=1).clamp_(min=self.scales) # (b)
            loss = []
            for b in range(targets_cls.shape[0]):
                cls_out_b = cls_out[b][mask_cls[b]] # (S+-, classes)
                reg_out_b = reg_out[b][mask_reg[b]] # (S+, 4)
                targets_cls_b = targets_cls[b][mask_cls[b]] # (S+-)
                targets_reg_b = targets_reg[b][mask_reg[b]] # # (S+, 4)
                loss_cls_b = sigmoid_focal_loss(cls_out_b, targets_cls_b, 2.0, 0.25).sum().view(1)
                loss_reg_b = smooth_l1_loss(reg_out_b, targets_reg_b, 0.11).sum().view(1)
                loss.append((loss_cls_b + loss_reg_b) / float(num_pos[b])) 
            return torch.cat(loss, dim=0) # (b)
        else:
            return self._decode(cls_out, reg_out, loc)


    def _encode(self, label_class, label_box, loc):
        '''
        Param:
        label_class: LongTensor(batch_num, N_max)
        label_box:   FloatTensor(batch_num, N_max, 4)
        loc:         FloatTensor(batch_num, 4)

        Return:
        targets_cls: LongTensor(batch_num, hwan)
        targets_reg: FloatTensor(batch_num, hwan, 4)
        '''
        targets_cls, targets_reg = [], []
        for b in range(label_class.shape[0]):
            targets_cls_b = torch.full((self.view_hwan,), -1,
                    dtype=torch.long, device=label_class.device)
            targets_reg_b = torch.zeros(self.view_hwan, 4, 
                    dtype=torch.float, device=label_class.device)
            iou = box_iou(self.view_anchors_yxyx, label_box[b]) # (hwan, N)
            if (iou.shape[1] <= 0):
                targets_cls_b[:] = 0
                targets_cls.append(targets_cls_b)
                targets_reg.append(targets_reg_b)
                continue
            iou_max, iou_max_idx = torch.max(iou, dim=1) # (hwan), (hwan)
            anchors_pos_mask = iou_max >= self.iou_th[1] # (hwan)
            anchors_neg_mask = iou_max < self.iou_th[0] # (hwan)
            # neg
            targets_cls_b[anchors_neg_mask] = 0
            # pos
            label_select = iou_max_idx[anchors_pos_mask]
            targets_cls_b[anchors_pos_mask] = label_class[b][label_select]
            # pos-reg
            lb_yxyx = label_box[b][label_select] # [S, 4]
            # Method 1
            d_yxyx = lb_yxyx - self.view_anchors_yxyx[anchors_pos_mask] # (S, 4)
            anchors_hw = self.view_anchors_hw[anchors_pos_mask]
            d_yxyx[:, :2] = d_yxyx[:, :2] / anchors_hw / 0.2
            d_yxyx[:, 2:] = d_yxyx[:, 2:] / anchors_hw / 0.2
            targets_reg_b[anchors_pos_mask] = d_yxyx
            # Method 2
            # lb_yx = (lb_yxyx[:, :2] + lb_yxyx[:, 2:]) / 2.0
            # lb_hw = lb_yxyx[:, 2:] - lb_yxyx[:, :2] + 1
            # a_yx = self.view_anchors_yx[anchors_pos_mask]
            # a_hw = self.view_anchors_hw[anchors_pos_mask]
            # pred_yx = (lb_yx - a_yx) / a_hw
            # pred_hw = torch.log(lb_hw / a_hw)
            # targets_reg_b[anchors_pos_mask] = torch.cat([pred_yx / 0.1, pred_hw / 0.2], dim=1)
            # ignore
            cd1 = self.view_anchors_yx - loc[b, :2]
            cd2 = loc[b, 2:] - self.view_anchors_yx
            mask = (cd1.min(dim=1)[0] < 0) | (cd2.min(dim=1)[0] < 0)
            targets_cls_b[mask] = -1
            # append
            targets_cls.append(targets_cls_b)
            targets_reg.append(targets_reg_b)
        return torch.stack(targets_cls), torch.stack(targets_reg)
    

    def _decode(self, cls_out, reg_out, loc):
        '''
        Param:
        cls_out: FloatTensor(batch_num, hwan, classes)
        reg_out: FloatTensor(batch_num, hwan, 4)
        loc:     FloatTensor(batch_num, 4)
        
        Return:
        cls_i_preds: LongTensor(batch_num, topk)
        cls_p_preds: FloatTensor(batch_num, topk)
        reg_preds:   FloatTensor(batch_num, topk, 4)
        '''
        cls_p_preds, cls_i_preds = torch.max(cls_out.sigmoid(), dim=2)
        cls_i_preds = cls_i_preds + 1
        reg_preds = []
        for b in range(cls_out.shape[0]):
            # box transform
            # Method 1
            reg_dyxyx = reg_out[b]
            reg_dyxyx[:, :2] = reg_dyxyx[:, :2] * 0.2 * self.view_anchors_hw
            reg_dyxyx[:, 2:] = reg_dyxyx[:, 2:] * 0.2 * self.view_anchors_hw
            reg_yxyx = reg_dyxyx + self.view_anchors_yxyx
            reg_preds.append(reg_yxyx)
            # Method 2
            # reg_pyxhw = reg_out[b]
            # lb_yx = reg_pyxhw[:, :2]  * 0.1 * self.view_anchors_hw + self.view_anchors_yx
            # lb_hw = (reg_pyxhw[:, 2:] * 0.2).exp() * self.view_anchors_hw
            # lb_ymin_xmin = lb_yx - lb_hw / 2.0
            # lb_ymax_xmax = lb_yx + lb_hw / 2.0 - 1
            # reg_preds.append(torch.cat([lb_ymin_xmin, lb_ymax_xmax], dim=1))
            # ignore
            cd1 = self.view_anchors_yx - loc[b, :2]
            cd2 = loc[b, 2:] - self.view_anchors_yx
            mask = (cd1.min(dim=1)[0] < 0) | (cd2.min(dim=1)[0] < 0)
            cls_p_preds[b, mask] = 0
        reg_preds = torch.stack(reg_preds)
        # topk
        nms_maxnum = min(int(self.max_detections), int(cls_p_preds.shape[1]))
        select = torch.topk(cls_p_preds, nms_maxnum, largest=True, dim=1)[1]
        _cls_i, _cls_p, _reg = [], [], []
        for b in range(cls_out.shape[0]):
            _cls_i.append(cls_i_preds[b][select[b]]) # (topk)
            _cls_p.append(cls_p_preds[b][select[b]]) # (topk)
            reg_preds_b = reg_preds[b][select[b]] # (topk, 4)
            reg_preds_b[:, 0].clamp_(min=float(loc[b, 0]))
            reg_preds_b[:, 1].clamp_(min=float(loc[b, 1]))
            reg_preds_b[:, 2].clamp_(max=float(loc[b, 2]))
            reg_preds_b[:, 3].clamp_(max=float(loc[b, 3]))
            _reg.append(reg_preds_b) # (topk, 4)
        return torch.stack(_cls_i), torch.stack(_cls_p), torch.stack(_reg)



def get_loss(temp):
    return torch.mean(temp)



def get_pred(temp, nms_th, nms_iou):
    '''
    temp:
    cls_i_preds: LongTensor(batch_num, topk)
    cls_p_preds: FloatTensor(batch_num, topk)
    reg_preds:   FloatTensor(batch_num, topk, 4)

    Return:
    cls_i_preds: (LongTensor(s1), LongTensor(s2), ...)
    cls_p_preds: (FloatTensor(s1), FloatTensor(s2), ...)
    reg_preds:   (FloatTensor(s1,4), FloatTensor(s2,4), ...)
    '''
    cls_i_preds, cls_p_preds, reg_preds = temp
    _cls_i_preds, _cls_p_preds, _reg_preds = [], [], []
    for b in range(cls_i_preds.shape[0]):
        cls_i_preds_b = cls_i_preds[b]
        cls_p_preds_b = cls_p_preds[b]
        reg_preds_b = reg_preds[b]
        mask = cls_p_preds_b > nms_th
        cls_i_preds_b = cls_i_preds_b[mask]
        cls_p_preds_b = cls_p_preds_b[mask]
        reg_preds_b = reg_preds_b[mask]
        keep = box_nms(reg_preds_b, cls_p_preds_b, nms_iou)
        _cls_i_preds.append(cls_i_preds_b[keep])
        _cls_p_preds.append(cls_p_preds_b[keep])
        _reg_preds.append(reg_preds_b[keep])
    return _cls_i_preds, _cls_p_preds, _reg_preds
