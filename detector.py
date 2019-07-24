import math
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from utils_box.anchors import gen_anchors, box_iou
from libs.sigmoid_focal_loss import sigmoid_focal_loss
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
        self.train_size = 641
        self.eval_size = 641
        self.iou_th = (0.3, 0.5)
        self.classes = 20
        self.nms_th = 0.05
        self.nms_iou = 0.5
        self.max_detections = 1000
        # ---------------------------

        # structure =======================================================
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

        # initialize =======================================================
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
        self._train_anchors_yxyx, self.train_anchors_yxhw = \
            gen_anchors(self.a_hw, self.scales, self.train_size, self.first_stride)
        self.train_an = self.train_anchors_yxhw.shape[0]
        self._train_ay_ax = self.train_anchors_yxhw[:, :2]
        self._train_ah_aw = self.train_anchors_yxhw[:, 2:]

        self._eval_anchors_yxyx, self.eval_anchors_yxhw = \
            gen_anchors(self.a_hw, self.scales, self.eval_size, self.first_stride)
        self.eval_an = self.eval_anchors_yxhw.shape[0]
        self._eval_ay_ax = self.eval_anchors_yxhw[:, :2]
        self._eval_ah_aw = self.eval_anchors_yxhw[:, 2:]

        self.register_buffer('train_anchors_yxyx', self._train_anchors_yxyx)
        self.register_buffer('train_ay_ax', self._train_ay_ax)
        self.register_buffer('train_ah_aw', self._train_ah_aw)

        self.register_buffer('eval_anchors_yxyx', self._eval_anchors_yxyx)
        self.register_buffer('eval_ay_ax', self._eval_ay_ax)
        self.register_buffer('eval_ah_aw', self._eval_ah_aw)


    def upsample(self, input):
        '''
        ATTENTION: size must be odd
        '''
        return F.interpolate(input, size=(input.shape[2]*2-1, input.shape[3]*2-1),
                    mode='bilinear', align_corners=True)
    

    def forward(self, x, label_class=None, label_box=None):
        '''
        Param:
        label_class: LongTensor(batch_num, N_max) or None
        label_box:   FloatTensor(batch_num, N_max, 4) or None

        Return:
        if label_class == None:
        tuple(
        loss_cls: FloatTensor(1), # sum
        loss_reg: FloatTensor(1), # sum
        num_pos:  FloatTensor(1), # sum
        )
        else:
        tuple(
        cls_i_preds: LongTensor(batch_num, topk)
        cls_p_preds: FloatTensor(batch_num, topk)
        reg_preds:   FloatTensor(batch_num, topk, 4)
        )
        '''
        
        C3, C4, C5 = self.backbone(x)
        
        P5 = self.prj_5(C5)
        P5_upsampled = self.upsample(P5)
        P5 = self.conv_5(P5)

        P4 = self.prj_4(C4)
        P4 = P5_upsampled + P4
        P4_upsampled = self.upsample(P4)
        P4 = self.conv_4(P4)

        P3 = self.prj_3(C3)
        P3 = P4_upsampled + P3
        P3 = self.conv_3(P3)

        P6 = self.conv_out6(C5)
        P7 = self.conv_out7(self.relu(P6))

        pred_list = [P3, P4, P5, P6, P7]

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
        
        # cls_out[b, sum_scale(Hi*Wi*an), classes]
        # reg_out[b, sum_scale(Hi*Wi*an), 4]
        cls_out = torch.cat(cls_out, dim=1)
        reg_out = torch.cat(reg_out, dim=1)

        if (label_class is None) or (label_box is None):
            cls_i_preds, cls_p_preds, reg_preds = self._decode(cls_out, reg_out)
            return (cls_i_preds, cls_p_preds, reg_preds)
        else:
            targets_cls, targets_reg = self._encode(label_class, label_box)
            mask_cls = targets_cls > -1 # (b, an)
            cls_out = cls_out[mask_cls] # (S+-, classes)
            reg_out = reg_out[mask_cls] # (S+-, 4)
            targets_cls = targets_cls[mask_cls] # (S+-)
            targets_reg = targets_reg[mask_cls] # (S+-, 4)
            loss_cls_1 = sigmoid_focal_loss(cls_out, targets_cls, 2.0, 0.25) # (S+-, classes)
            loss_cls_1 = torch.sum(loss_cls_1).view(1)
            mask_reg = targets_cls > 0 # (S+)
            reg_out = reg_out[mask_reg] # (S+, 4)
            targets_reg = targets_reg[mask_reg] # # (S+, 4)
            loss_reg = F.smooth_l1_loss(reg_out, targets_reg, reduction='sum').view(1)
            num_pos = torch.sum(mask_reg).view(1)
            return (loss_cls_1, loss_reg, num_pos)
    

    def _encode(self, label_class, label_box):
        '''
        Param:
        label_class: LongTensor(batch_num, N_max)
        label_box:   FloatTensor(batch_num, N_max, 4)

        Return:
        targets_cls: LongTensor(batch_num, an)
        targets_reg: FloatTensor(batch_num, an, 4)
        '''
        label_class_out = []
        label_box_out   = []

        for b in range(label_class.shape[0]):

            label_class_out_b = torch.full((self.train_an,), -1).long().to(label_class.device)
            label_box_out_b = torch.zeros(self.train_an, 4).to(label_class.device)
            iou = box_iou(self.train_anchors_yxyx, label_box[b]) # [an, N]
            
            if (iou.shape[1] <= 0):
                # print('Find an image that does not contain objects')
                label_class_out_b[:] = 0
                label_class_out.append(label_class_out_b)
                label_box_out.append(label_box_out_b)
                continue
            
            ############################==============================
            iou_pos_mask = iou > self.iou_th[1] # [an, Nb]
            iou_neg_mask = iou < self.iou_th[0] # [an, Nb]
            label_select = torch.argmax(iou, dim=1)   # [an]
            anchors_select = torch.argmax(iou, dim=0) # [Nb]
            anchors_pos_mask = torch.max(iou_pos_mask, dim=1)[0].byte() # [an]
            anchors_neg_mask = torch.min(iou_neg_mask, dim=1)[0].byte() # [an]
            # get class targets background
            label_class_out_b[anchors_neg_mask] = 0
            # get class targets 2
            label_class_out_b[anchors_select] = label_class[b]
            # get class targets 1
            label_select_1 = label_select[anchors_pos_mask]
            label_class_out_b[anchors_pos_mask] = label_class[b][label_select_1]
            # get box targets 2
            lb_yxyx_2 = label_box[b] # [Nb, 4]
            ay_ax = self.train_ay_ax[anchors_select]
            ah_aw = self.train_ah_aw[anchors_select]
            lb_ymin_xmin_2, lb_ymax_xmax_2 = lb_yxyx_2[:, :2], lb_yxyx_2[:, 2:]
            lbh_lbw_2 = lb_ymax_xmax_2 - lb_ymin_xmin_2
            lby_lbx_2 = (lb_ymin_xmin_2 + lb_ymax_xmax_2)/2.0
            f1_f2_2 = (lby_lbx_2 - ay_ax) / ah_aw
            f3_f4_2 = (lbh_lbw_2 / ah_aw + 1e-10).log()
            label_box_out_b[anchors_select] = torch.cat([f1_f2_2, f3_f4_2], dim=1)
            # get box targets 1
            lb_yxyx_1 = label_box[b][label_select_1] # [S, 4]
            ay_ax = self.train_ay_ax[anchors_pos_mask]
            ah_aw = self.train_ah_aw[anchors_pos_mask]
            lb_ymin_xmin_1, lb_ymax_xmax_1 = lb_yxyx_1[:, :2], lb_yxyx_1[:, 2:]
            lbh_lbw_1 = lb_ymax_xmax_1 - lb_ymin_xmin_1
            lby_lbx_1 = (lb_ymin_xmin_1 + lb_ymax_xmax_1)/2.0
            f1_f2_1 = (lby_lbx_1 - ay_ax) / ah_aw
            f3_f4_1 = (lbh_lbw_1 / ah_aw + 1e-10).log()
            label_box_out_b[anchors_pos_mask] = torch.cat([f1_f2_1, f3_f4_1], dim=1)
            ############################==============================
            
            label_class_out.append(label_class_out_b)
            label_box_out.append(label_box_out_b)

        targets_cls = torch.stack(label_class_out, dim=0)
        targets_reg = torch.stack(label_box_out, dim=0)
        return targets_cls, targets_reg
    

    def _decode(self, cls_out, reg_out):
        '''
        Param:
        cls_out: FloatTensor(batch_num, an, classes)
        reg_out: FloatTensor(batch_num, an, 4)
        
        Return:
        cls_i_preds: LongTensor(batch_num, topk)
        cls_p_preds: FloatTensor(batch_num, topk)
        reg_preds:   FloatTensor(batch_num, topk, 4)
        '''

        cls_p_preds, cls_i_preds = torch.max(cls_out.sigmoid(), dim=2)
        cls_i_preds = cls_i_preds + 1
        reg_preds = []
        for b in range(cls_out.shape[0]):
            f1_f2 = reg_out[b, :, :2]
            f3_f4 = reg_out[b, :, 2:]
            y_x = f1_f2 * self.eval_ah_aw + self.eval_ay_ax
            h_w = f3_f4.exp() * self.eval_ah_aw
            ymin_xmin = y_x - h_w/2.0
            ymax_xmax = y_x + h_w/2.0
            ymin_xmin_ymax_xmax = torch.cat([ymin_xmin, ymax_xmax], dim=1)
            reg_preds.append(ymin_xmin_ymax_xmax)
        reg_preds = torch.stack(reg_preds, dim=0)
        
        # Topk
        nms_maxnum = min(int(self.max_detections), int(cls_p_preds.shape[1]))
        select = torch.topk(cls_p_preds, nms_maxnum, largest=True, dim=1)[1]
        
        # NMS
        list_cls_i_preds = []
        list_cls_p_preds = []
        list_reg_preds = []

        for b in range(cls_out.shape[0]):
            cls_i_preds_b = cls_i_preds[b][select[b]] # (topk)
            cls_p_preds_b = cls_p_preds[b][select[b]] # (topk)
            reg_preds_b = reg_preds[b][select[b]] # (topk, 4)
            list_cls_i_preds.append(cls_i_preds_b)
            list_cls_p_preds.append(cls_p_preds_b)
            list_reg_preds.append(reg_preds_b)
            
        return torch.stack(list_cls_i_preds, dim=0), \
                    torch.stack(list_cls_p_preds, dim=0), \
                        torch.stack(list_reg_preds, dim=0)



def get_loss(temp):
    loss_cls_1, loss_reg, num_pos = temp
    loss_cls = torch.sum(loss_cls_1)
    loss_reg = torch.sum(loss_reg)
    num_pos = float(torch.sum(num_pos))
    if num_pos <= 0:
        num_pos = 1.0
    loss = (loss_cls + loss_reg) / num_pos
    return loss



def get_pred(temp, nms_th, nms_iou, eval_size):
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

    list_cls_i_preds = []
    list_cls_p_preds = []
    list_reg_preds = []

    for b in range(cls_i_preds.shape[0]):

        cls_i_preds_b = cls_i_preds[b]
        cls_p_preds_b = cls_p_preds[b]
        reg_preds_b = reg_preds[b]
        
        mask = cls_p_preds_b > nms_th
        cls_i_preds_b = cls_i_preds_b[mask]
        cls_p_preds_b = cls_p_preds_b[mask]
        reg_preds_b = reg_preds_b[mask]

        keep = box_nms(reg_preds_b, cls_p_preds_b, nms_iou)
        cls_i_preds_b = cls_i_preds_b[keep]
        cls_p_preds_b = cls_p_preds_b[keep]
        reg_preds_b = reg_preds_b[keep]

        reg_preds_b[:, :2] = reg_preds_b[:, :2].clamp(min=0)
        reg_preds_b[:, 2:4] = reg_preds_b[:, 2:4].clamp(max=eval_size-1)

        list_cls_i_preds.append(cls_i_preds_b)
        list_cls_p_preds.append(cls_p_preds_b)
        list_reg_preds.append(reg_preds_b)
    
    return list_cls_i_preds, list_cls_p_preds, list_reg_preds
