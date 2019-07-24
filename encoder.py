import torch
from utils_box.anchors import gen_anchors
from libs.nms import box_nms 
# TODO: define Encoder



def box_iou(box1, box2, eps=1e-10):
    '''
    Param:
    box1:   FloatTensor(n,4) # 4: ymin, xmin, ymax, xmax
    box2:   FloatTensor(m,4)

    Return:
    FloatTensor(n,m)
    '''
    tl = torch.max(box1[:,None,:2], box2[:,:2])  # [n,m,2]
    br = torch.min(box1[:,None,2:], box2[:,2:])  # [n,m,2]
    hw = (br-tl+eps).clamp(min=0)  # [n,m,2]
    inter = hw[:,:,0] * hw[:,:,1]  # [n,m]
    area1 = (box1[:,2]-box1[:,0]+eps) * (box1[:,3]-box1[:,1]+eps)  # [n,]
    area2 = (box2[:,2]-box2[:,0]+eps) * (box2[:,3]-box2[:,1]+eps)  # [m,]
    iou = inter / (area1[:,None] + area2 - inter)
    return iou



class Encoder:
    def __init__(self, detector):

        self.a_hw = detector.a_hw
        self.scales = detector.scales
        self.first_stride = detector.first_stride
        self.train_iou_th = detector.train_iou_th
        self.train_size = detector.train_size
        self.eval_size = detector.eval_size
        self.nms = detector.nms
        self.nms_th = detector.nms_th
        self.nms_iou = detector.nms_iou
        self.max_detections = detector.max_detections

        self.train_anchors_yxyx, self.train_anchors_yxhw = \
            gen_anchors(self.a_hw, self.scales, self.train_size, self.first_stride)
        self.train_an = self.train_anchors_yxyx.shape[0]
        self.train_ay_ax = self.train_anchors_yxhw[:, :2]
        self.train_ah_aw = self.train_anchors_yxhw[:, 2:]

        self.eval_anchors_yxyx, self.eval_anchors_yxhw = \
            gen_anchors(self.a_hw, self.scales, self.eval_size, self.first_stride)
        self.eval_an = self.eval_anchors_yxyx.shape[0]
        self.eval_ay_ax = self.eval_anchors_yxhw[:, :2]
        self.eval_ah_aw = self.eval_anchors_yxhw[:, 2:]

    def encode(self, label_class, label_box):
        '''
        Param:
        label_class: (LongTensor(N1), LongTensor(N2), ...)
        label_box:   (FloatTensor(N1,4), FloatTensor(N2,4), ...)

        Return:
        class_targets: LongTensor(batch_num, an)
        box_targets:   FloatTensor(batch_num, an, 4)

        Note:
        - class = 0 indicate background
        - in label_box 4 indicate ymin, xmin, ymax, xmax
        - all calculations are on the CPU
        - an: acc_scale(Hi*Wi*len(a_hw))
          Hi,Wi accumulate from big to small
        - in class_targets, 0:background; -1:exclude; n(>0):class_idx
        - in box_targets, 4 indicates:
            f1 -> (lb_y - a_y) / a_h
            f2 -> (lb_x - a_x) / a_w
            f3 -> log(lb_h / a_h)
            f4 -> log(lb_w / a_w)
        '''
        label_class_out = []
        label_box_out   = []

        for b in range(len(label_class)):

            label_class_out_b = torch.full((self.train_an,), -1).long()
            label_box_out_b = torch.zeros(self.train_an, 4)
            iou = box_iou(self.train_anchors_yxyx, label_box[b]) # [an, Nb]
            
            if (iou.shape[1] <= 0):
                # print('Find an image that does not contain objects')
                label_class_out_b[:] = 0
                label_class_out.append(label_class_out_b)
                label_box_out.append(label_box_out_b)
                continue
            
            ############################可优化项==============================
            iou_pos_mask = iou > self.train_iou_th[1] # [an, Nb]
            iou_neg_mask = iou < self.train_iou_th[0] # [an, Nb]
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
            ############################可优化项==============================
            
            label_class_out.append(label_class_out_b)
            label_box_out.append(label_box_out_b)

        class_targets = torch.stack(label_class_out, dim=0)
        box_targets = torch.stack(label_box_out, dim=0)
        return (class_targets, box_targets)

    def decode(self, net_out, scale_shift=None):
        '''
        Param:
        cls_out: FloatTensor(batch_num, an, classes)
        reg_out: FloatTensor(batch_num, an, 4)
        
        Return:
        if nms:
            cls_i_preds: (LongTensor(s1), LongTensor(s2), ...)
            cls_p_preds: (FloatTensor(s1), FloatTensor(s2), ...)
            reg_preds:   (FloatTensor(s1,4), FloatTensor(s2,4), ...)
        else:
            cls_i_preds: LongTensor(batch_num, an)
            cls_p_preds: FloatTensor(batch_num, an)
            reg_preds:   FloatTensor(batch_num, an, 4)

        Note:
        - scale_shift: if not None, reg_preds /= float(scale_shift)
        - class = 0 indicate background
        - in reg_preds 4 indicate ymin, xmin, ymax, xmax
        - all calculations are on the CPU
        - an: acc_scale(Hi*Wi*len(a_hw))
          Hi,Wi accumulate from big to small
        - reg_out = f1, f2, f3, f4, decoding process:
            y = f1 * a_h + a_y
            x = f2 * a_w + a_x
            h = f3.exp() * a_h
            w = f4.exp() * a_w
            ymin, xmin = y-h/2, x-w/2
            ymax, xmax = y+h/2, x+w/2
        '''
        cls_out, reg_out = net_out

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

        if self.nms == False:
            if scale_shift is not None:
                reg_preds /= float(scale_shift)
            return cls_i_preds, cls_p_preds, reg_preds
        
        # Topk
        nms_maxnum = min(int(self.max_detections), int(cls_p_preds.shape[1]))
        select = torch.topk(cls_p_preds, nms_maxnum, largest=True, dim=1)[1]

        # NMS
        _cls_i_preds = []
        _cls_p_preds = []
        _reg_preds = []

        for b in range(cls_out.shape[0]):

            cls_i_preds_b = cls_i_preds[b][select[b]] # (topk)
            cls_p_preds_b = cls_p_preds[b][select[b]] # (topk)
            reg_preds_b = reg_preds[b][select[b]] # (topk, 4)

            mask = cls_p_preds_b > self.nms_th
            cls_i_preds_b = cls_i_preds_b[mask]
            cls_p_preds_b = cls_p_preds_b[mask]
            reg_preds_b = reg_preds_b[mask]

            keep = box_nms(reg_preds_b, cls_p_preds_b, self.nms_iou)
            cls_i_preds_b = cls_i_preds_b[keep]
            cls_p_preds_b = cls_p_preds_b[keep]
            reg_preds_b = reg_preds_b[keep]

            reg_preds_b[:, :2] = reg_preds_b[:, :2].clamp(min=0)
            reg_preds_b[:, 2] = reg_preds_b[:, 2].clamp(max=self.eval_size-1)
            reg_preds_b[:, 3] = reg_preds_b[:, 3].clamp(max=self.eval_size-1)

            if scale_shift is not None:
                reg_preds_b /= float(scale_shift)

            _cls_i_preds.append(cls_i_preds_b)
            _cls_p_preds.append(cls_p_preds_b)
            _reg_preds.append(reg_preds_b)
            
        return _cls_i_preds, _cls_p_preds, _reg_preds
