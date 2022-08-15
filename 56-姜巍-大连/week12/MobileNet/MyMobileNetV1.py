import torch
import torch.nn as nn
import torch.nn.functional as F


class DSC(nn.Module):
    """Define a depthwise separable convolution sequence"""

    def __init__(self, i_features, o_features, stride_shape=1):
        super().__init__()

        self.dw_conv = nn.Sequential(
            # Depthwise convolution(Conv dw), 使用group参数实现普通卷积转化成depthwise conv。
            nn.Conv2d(i_features, i_features, 3, stride_shape, padding=1, groups=i_features),
            nn.BatchNorm2d(i_features),
            nn.ReLU6(),
            # Pointwise convolution(1 × 1 convolution)
            nn.Conv2d(i_features, o_features, 1),
            nn.BatchNorm2d(o_features),
            nn.ReLU6()
        )

    def forward(self, x):
        """前馈函数"""

        return self.dw_conv(x)


class ConvBNReLu(nn.Module):
    """Define a sequence of convolution-BatchNormalization-ReLu"""

    def __init__(self, i_features, o_features, stride_shape=1):
        super().__init__()

        self.cbr = nn.Sequential(
            nn.Conv2d(i_features, o_features, 3, stride_shape, padding=1),
            nn.BatchNorm2d(o_features),
            nn.ReLU6()
        )

    def forward(self, x):
        """前馈函数"""

        return self.cbr(x)


class MyMobileNetV1(nn.Module):
    """Define a mobile net V1"""

    def __init__(self, in_features=3):
        super().__init__()

        self.net = nn.Sequential(
            # input size: B × 3 × 224 × 224
            ConvBNReLu(in_features, 32, stride_shape=2),
            # B × 32 × 112 × 112
            DSC(32, 64),
            # B × 64 × 112 × 112
            DSC(64, 128, stride_shape=2),
            # B × 128 × 56 × 56
            DSC(128, 128),
            # B × 128 × 56 × 56
            DSC(128, 256, stride_shape=2),
            # B × 256 × 28 × 28
            DSC(256, 256),
            # B × 256 × 28 × 28
            DSC(256, 512, stride_shape=2),
            # B × 512 × 14 × 14
            DSC(512, 512),
            DSC(512, 512),
            DSC(512, 512),
            DSC(512, 512),
            DSC(512, 512),
            DSC(512, 1024, stride_shape=2),
            # B × 1024 × 7 × 7
            DSC(1024, 1024, stride_shape=1),  # [注：此处与论文中 Table 1 列出的stride=2不同，怀疑原文此处是个未校正的错误]
            # B × 1024 × 7 × 7
            nn.AdaptiveAvgPool2d((1, 1)),
            # out: B × 1024 × 1 × 1
        )

        self.fc1 = nn.Linear(1024, 1000)

    def forward(self, x):
        """前馈函数"""

        x = self.net(x)
        x = x.view(-1, 1024)  # 输出为1024列。1024为线性层的输入数值
        x = self.fc1(x)

        return x
