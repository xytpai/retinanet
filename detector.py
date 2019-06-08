import math
import torch 
import torch.nn as nn 
import torch.nn.functional as F 



class Detector(nn.Module):
    def __init__(self, classes=20, anchors=3):
        super(Detector, self).__init__()
        # 每个类别一个输出, 这里不引入背景
        # 如果为背景那么这些类别对应的输出单元值都是0
        # 如果为一种类别的物体那么只有那个类对项为1其余为0
        # 最后使用逐点Sigmoid后使用类似BCE的FocalLoss损失函数输出
        self.classes = classes
        self.relu = nn.ReLU(inplace=True)
        # s6级预测是在s5的输出特征基础上再做一次3X3卷积
        # 检测器的所有卷积层都不使用BN
        self.conv_out6 = nn.Conv2d(2048, 256, kernel_size=3, padding=1, stride=2)
        self.conv_out7 = nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=2)
        # s5级预测是在s5的输出特征上执行一个降维操作
        self.prj_5 = nn.Conv2d(2048, 256, kernel_size=1)
        # 下面的prj_4,3是为了将out4,3映射以执行加法操作
        self.prj_4 = nn.Conv2d(1024, 256, kernel_size=1)
        self.prj_3 = nn.Conv2d(512, 256, kernel_size=1)
        # 以下两个卷积只对融合过的s4,s3级特征施加
        # 为了减少由于上采样产生的不连续性
        self.conv_4 =nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv_3 =nn.Conv2d(256, 256, kernel_size=3, padding=1)
        # 分类器, 所有级预测都共享一个分类器
        self.conv_cls = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, anchors*classes, kernel_size=3, padding=1))
        # 目标框回归器, 所有级预测都共享一个回归器
        self.conv_reg = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, anchors*4, kernel_size=3, padding=1))
        # 除了分类器最后一层其余都初始化为 weight:std=0.01, bias:0
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.constant_(layer.bias, 0)
                nn.init.normal_(layer.weight, mean=0, std=0.01)
        # 分类器最后一层的 weight 需要特殊初始化
        # 代表在训练开始时, 每一个锚点输出都为前景
        pi = 0.01
        _bias = -math.log((1.0-pi)/pi)
        nn.init.constant_(self.conv_cls[-1].bias, _bias)

    def upsample(self, input):
        # 为了增加精度使用双线性插值
        # 由于本实验输入图像尺寸固定为2^n+1因此可以计算插值系数
        # 这种情况下 align_corners=True 代表四角对其
        return F.interpolate(input, size=(input.shape[2]*2-1, input.shape[3]*2-1),
                    mode='bilinear', align_corners=True)

    def forward(self, out_list):
        # 拆分输入, 这个输入就是前面的 Resnet-50 输出
        out3, out4, out5 = out_list
        # 直接卷积out5得到s6预测层
        out6_pred = self.conv_out6(out5)
        out7_pred = self.conv_out7(out6_pred)
        # 直接投影out5得到s5预测层
        out5_pred = self.prj_5(out5)
        # s4预测层:s5预测层上采用+out4投影
        out5_up = self.upsample(out5_pred)
        out4_pred = out5_up + self.prj_4(out4)
        # s3预测层:s4预测层上采用+out3投影
        out4_up = self.upsample(out4_pred)
        out3_pred = out4_up + self.prj_3(out3)
        # 减少不连续性，增加一致性
        out7_pred = self.relu(out7_pred)
        out6_pred = self.relu(out6_pred)
        out5_pred = self.relu(out5_pred)
        out4_pred = self.relu(self.conv_4(out4_pred))
        out3_pred = self.relu(self.conv_3(out3_pred))
        # 按照步级从小到大排列
        pred_list = [out3_pred, out4_pred, out5_pred, out6_pred, out7_pred]
        # 获得最终输出
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
        # 最终输出 cls_out, reg_out
        # cls_out[b, sum_scale(Hi*Wi*an), classes]
        # cls_reg[b, sum_scale(Hi*Wi*an), 4]
        # 其中 sum_scale 中按照特征从大到小/步级从小到大排列
        return torch.cat(cls_out, dim=1), torch.cat(reg_out, dim=1)
        