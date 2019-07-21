import math
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from libs.sigmoid_focal_loss import sigmoid_focal_loss
# TODO: choose backbone
from backbone import resnet50 as backbone



class Detector(nn.Module):
    def __init__(self, pretrained=False):
        '''
        Return:
        cls_out: FloatTensor(b, sum_scale(Hi*Wi*an), classes)
        reg_out: FloatTensor(b, sum_scale(Hi*Wi*an), 4)

        Note:
        sum_scale(): [P(i), P(i+1), P(i+2), ...]
        '''
        super(Detector, self).__init__()

        # ---------------------------
        # TODO: Param
        self.a_hw = [
            [32.00, 32.00],
            [35.92, 35.92],
            [40.32, 40.32],

            [22.63, 45.25],
            [25.40, 50.80],
            [28.51, 57.01],

            [45.25, 22.63],
            [50.80, 25.40],
            [57.01, 28.51],
        ]
        self.scales = 5
        self.first_stride = 8
        self.train_size = 1025
        self.eval_size = 1025
        self.iou_th = (0.4, 0.5)
        self.classes = 80
        self.nms = True
        self.nms_th = 0.05
        self.nms_iou = 0.5
        self.max_detections = 1000
        # ---------------------------

        self.backbone = backbone(pretrained=pretrained)
        self.anchors = len(self.a_hw)

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
            nn.Conv2d(256, self.anchors*self.classes, kernel_size=3, padding=1))
        self.conv_reg = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, self.anchors*4, kernel_size=3, padding=1))

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


    def upsample(self, input):
        '''
        ATTENTION: size must be odd
        '''
        return F.interpolate(input, size=(input.shape[2]*2-1, input.shape[3]*2-1),
                    mode='bilinear', align_corners=True)
    

    def forward(self, x, targets_cls=None, targets_reg=None):
        '''
        targets_cls: LongTensor(b, an)
        targets_reg: FloatTensor(b, an, 4)
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

        if targets_cls is None:
            return cls_out, reg_out
        else:
            mask_cls = targets_cls > -1 # (b, an)
            cls_out = cls_out[mask_cls] # (S+-, classes)
            reg_out = reg_out[mask_cls] # (S+-, 4)
            targets_cls = targets_cls[mask_cls] # (S+-)
            targets_reg = targets_reg[mask_cls] # (S+-, 4)
            loss_cls_1 = sigmoid_focal_loss(cls_out, targets_cls, gamma=2.0, alpha=0.25)
            mask_reg = targets_cls > 0 # (S+)
            reg_out = reg_out[mask_reg] # (S+, 4)
            targets_reg = targets_reg[mask_reg] # # (S+, 4)
            return (loss_cls_1, mask_reg, reg_out, targets_reg)



def get_loss(temp):
    loss_cls_1, mask_reg, reg_out, targets_reg = temp
    loss_cls = torch.sum(loss_cls_1)
    num_pos = float(torch.sum(mask_reg))
    if num_pos <= 0:
        num_pos = 1.0
    loss_reg = F.smooth_l1_loss(reg_out, targets_reg, reduction='sum')
    loss = (loss_cls + loss_reg) / num_pos
    return loss
