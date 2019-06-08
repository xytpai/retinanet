import torch
import torch.nn as nn 
import torch.nn.functional as F



class Bottleneck(nn.Module):
    def __init__(self, inplanes, planes, outplanes, stride=1):
        super(Bottleneck, self).__init__()
        self.stride = stride
        self.projection = False 
        # 如果进行了下采样或者输入输出通道不同需要一个投影操作
        if (stride > 1) or (inplanes != outplanes):
            self.projection = True
        self.relu = nn.ReLU(inplace=True)
        self.conv_1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn_1 = nn.BatchNorm2d(planes)
        self.conv_2 = nn.Conv2d(planes, planes, kernel_size=3, 
            stride=stride, padding=1, bias=False)
        self.bn_2 = nn.BatchNorm2d(planes)
        self.conv_3 = nn.Conv2d(planes, outplanes, kernel_size=1, bias=False)        
        self.bn_3 = nn.BatchNorm2d(outplanes)
        if self.projection:
            self.conv_prj = nn.Conv2d(inplanes, outplanes, kernel_size=1, 
                stride=stride, bias=False)
            self.bn_prj = nn.BatchNorm2d(outplanes)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn_1(self.conv_1(x)))
        out = self.relu(self.bn_2(self.conv_2(out)))
        out = self.bn_3(self.conv_3(out))
        # 在非线性函数之前施加加法
        if self.projection: # 维度不同需要投影
            residual = self.bn_prj(self.conv_prj(x))
        out = self.relu(out + residual) # 注意加法不要InPlace操作
        return out



class Extractor(nn.Module):
    def __init__(self, classes=1000):
        super(Extractor, self).__init__()
        self.relu = nn.ReLU() # 这个ReLU只用一次而非InPlace操作
        self.conv_input = nn.Conv2d(3, 64, kernel_size=7, padding=3, 
            stride=2, bias=False)
        self.bn_input = nn.BatchNorm2d(64)
        self.max_pool = nn.MaxPool2d(kernel_size=3, padding=1, stride=2)
        # s4级一共3个残差单元
        self.resblock1_1 = Bottleneck(64, 64, 256, stride=1) # stride:4
        self.resblock1_2 = Bottleneck(256, 64, 256)
        self.resblock1_3 = Bottleneck(256, 64, 256)
        # s8级一共4个残差单元
        self.resblock2_1 = Bottleneck(256, 128, 512, stride=2) # stride:8
        self.resblock2_2 = Bottleneck(512, 128, 512)
        self.resblock2_3 = Bottleneck(512, 128, 512)
        self.resblock2_4 = Bottleneck(512, 128, 512)
        # s16级一共6个残差单元
        self.resblock3_1 = Bottleneck(512, 256, 1024, stride=2) # stride:16
        self.resblock3_2 = Bottleneck(1024, 256, 1024)
        self.resblock3_3 = Bottleneck(1024, 256, 1024)
        self.resblock3_4 = Bottleneck(1024, 256, 1024)
        self.resblock3_5 = Bottleneck(1024, 256, 1024)
        self.resblock3_6 = Bottleneck(1024, 256, 1024)
        # s32级一共3个残差单元
        self.resblock4_1 = Bottleneck(1024, 512, 2048, stride=2) # stride:32
        self.resblock4_2 = Bottleneck(2048, 512, 2048)
        self.resblock4_3 = Bottleneck(2048, 512, 2048)
        # 分类输出层用于做分类预训练
        self.fc = nn.Linear(2048, classes)

    def forward(self, x, classify=True):
        x = self.relu(self.bn_input(self.conv_input(x)))
        x = self.max_pool(x)
        # s4
        x = self.resblock1_1(x)
        x = self.resblock1_2(x)
        x = self.resblock1_3(x)
        # s8
        x = self.resblock2_1(x)
        x = self.resblock2_2(x)
        x = self.resblock2_3(x)
        out3 = self.resblock2_4(x)
        # s16
        x = self.resblock3_1(out3)
        x = self.resblock3_2(x)
        x = self.resblock3_3(x)
        x = self.resblock3_4(x)
        x = self.resblock3_5(x)
        out4 = self.resblock3_6(x)
        # s32
        x = self.resblock4_1(out4)
        x = self.resblock4_2(x)
        out5 = self.resblock4_3(x) # stride = 32
        # out3,4,5 分别代表 stride=2^3,2^4,2^5 的输出
        # out3 的特征尺寸是最大的, out5最小, 用于FPN跨层预测
        if classify: # 分类预训练开启
            H, W = out5.shape[2], out5.shape[3]
            out5 = F.avg_pool2d(out5, kernel_size=(H, W))
            out5 = out5.view(out5.shape[0], -1)
            out5 = self.fc(out5)
            return out5
        return (out3, out4, out5)

    def freeze_bn(self):
        # 目标检测一个Batch的图像数目较少
        # 由于使用预训练的特征提取器, BN层的均值与方差已经得到
        # 因此训练时直接固定住BN的均值和方差
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()


