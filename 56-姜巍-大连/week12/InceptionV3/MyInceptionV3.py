"""
InceptionV3论文正文缺少必要的网络结构详细信息，而包含这一部分信息的supplementary material却无从下载。
目前TensorFlow和PyTorch官网源码均与论文原文的网络结构有差异，原因尚不清楚。
本代码试图在不涉及辅助分类器功能部分的前提下，尽量按照原文编写InceptionV3网络结构
具体差异已经罗列在“Remark.md”文件中，请查阅。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNRuLu(nn.Module):
    """为了缩短代码，将一些基本的单元组合定义成一个类"""

    def __init__(self, i_channel: int, o_channel: int, kernel_shape: int | tuple, stride_shape: int | tuple = 1,
                 zero_padding_shape: int | tuple = 0, **kwargs):
        """
        初始化属性和网络结构
        :param i_channel: 输入通道数
        :param o_channel: 输出通道数
        :param kernel_shape: kernel的(h, w)
        :param stride_shape: stride(h, w)方向的步长
        :param zero_padding_shape: 0-padding(h, w)方向的层数
        :param kwargs: 其他关键字参数
        """

        super().__init__()

        self.cbr = nn.Sequential(
            nn.Conv2d(
                in_channels=i_channel,
                out_channels=o_channel,
                kernel_size=kernel_shape,
                stride=stride_shape,
                padding=zero_padding_shape,
                **kwargs
            ),
            nn.BatchNorm2d(num_features=o_channel, eps=0.001),  # eps=0.001来自官方文档定义
            nn.ReLU()
        )

    def forward(self, x):
        """前馈函数"""

        return self.cbr(x)


class InceptionFig5(nn.Module):
    """Inception modules where each 5 × 5 convolution is replaced by two 3 × 3 convolution"""

    def __init__(self, i_features, pool_features=64):
        """
        定义四个分支网络结构，初始化各层参数
        :param i_features: base/input features数量
        :param pool_features: average pooling分支输出feature map数量
        """

        super().__init__()

        # 1 × 1 convolution 分支
        self.branch_01 = ConvBNRuLu(
            i_channel=i_features,
            o_channel=64,
            kernel_shape=(1, 1)
        )

        # 3 × 3 convolution 分支
        self.branch_02 = nn.Sequential(
            ConvBNRuLu(i_features, 48, 1),
            ConvBNRuLu(48, 64, 3, zero_padding_shape=1)
        )

        # 5 × 5 (被两个3 × 3替代) convolution 分支
        self.branch_03 = nn.Sequential(
            ConvBNRuLu(i_features, 64, 1),
            ConvBNRuLu(64, 96, 3, zero_padding_shape=1),
            ConvBNRuLu(96, 96, 3, zero_padding_shape=1),
        )

        # average pooling 分支
        self.branch_04 = nn.Sequential(
            nn.AvgPool2d(3, 1, 1),
            ConvBNRuLu(i_features, pool_features, 1)
        )

    def forward(self, x):
        """前馈函数"""

        # 返回 Batch size × (64 + 64 + 96 + pool_features) × 35 × 35
        return torch.cat([self.branch_01(x), self.branch_02(x), self.branch_03(x), self.branch_04(x)], 1)


class InceptionFig5ToFig6(nn.Module):
    """Inception module that reduces the grid-size while expands the filter banks from InceptionFig5 to InceptionFig6"""

    def __init__(self, i_features):
        """
        定义三个分支网络结构，初始化各层参数，使feature map的网格尺寸降低(35 → 17)
        :param i_features: Module InceptionFig5 的输出结果作为输入
        """

        super().__init__()

        # 左侧分支: 3 × 3 (/2) convolution
        self.branch_left = ConvBNRuLu(i_features, 384, 3, 2)

        # 中间分支: 串联 1 × 1 and 3 × 3 and 3 × 3 (/2) convolution
        self.branch_middle = nn.Sequential(
            ConvBNRuLu(i_features, 64, 1),
            ConvBNRuLu(64, 96, 3, zero_padding_shape=1),
            ConvBNRuLu(96, 96, 3, 2)
        )

        # max pool 3 × 3 (/2)
        self.branch_right = nn.MaxPool2d(3, 2)

    def forward(self, x):
        """前馈函数，实现feature map网格尺寸降低"""

        # 返回 Batch size × 768 × 17 × 17
        return torch.cat([self.branch_left(x), self.branch_middle(x), self.branch_right(x)], 1)


class InceptionFig6(nn.Module):
    """Inception modules after the factorization of the 7 × 7 convolutions for the 17 × 17 grid"""

    def __init__(self, i_features, channels_inside=192):
        """
        定义四个分支网络结构，初始化各层参数
        :param i_features: base/input features数量
        :param channels_inside: 在module内部传递时的feature map数量
        """

        super().__init__()

        # 1 × 1 convolution 分支
        self.branch_01 = ConvBNRuLu(i_features, 192, 1)

        # 单层 7 × 7(串联 1 × 7 和 7 × 1) convolution 分支
        self.branch_02 = nn.Sequential(
            ConvBNRuLu(i_features, channels_inside, 1),
            ConvBNRuLu(channels_inside, channels_inside, (1, 7), zero_padding_shape=(0, 3)),
            ConvBNRuLu(channels_inside, 192, (7, 1), zero_padding_shape=(3, 0))
        )

        # 双层 7 × 7(串联 1 × 7 和 7 × 1 两次) convolution 分支
        self.branch_03 = nn.Sequential(
            ConvBNRuLu(i_features, channels_inside, 1),
            ConvBNRuLu(channels_inside, channels_inside, (1, 7), zero_padding_shape=(0, 3)),
            ConvBNRuLu(channels_inside, channels_inside, (7, 1), zero_padding_shape=(3, 0)),
            ConvBNRuLu(channels_inside, channels_inside, (1, 7), zero_padding_shape=(0, 3)),
            ConvBNRuLu(channels_inside, 192, (7, 1), zero_padding_shape=(3, 0))
        )

        # average pooling 分支
        self.branch_04 = nn.Sequential(
            nn.AvgPool2d(3, 1, 1),
            ConvBNRuLu(i_features, 192, 1)
        )

    def forward(self, x):
        """前馈函数"""

        # 返回 Batch size × 768 × 17 × 17
        return torch.cat([self.branch_01(x), self.branch_02(x), self.branch_03(x), self.branch_04(x)], 1)


class InceptionFig6ToFig7(nn.Module):
    """Inception module that reduces the grid-size while expands the filter banks from InceptionFig6 to InceptionFig7"""

    def __init__(self, i_features):
        """
        定义三个分支网络结构，初始化各层参数，使feature map的网格尺寸降低(17 → 8)
        :param i_features: base/input features数量
        """

        super().__init__()

        # 串联 1 × 1 and 3 × 3 (/2) convolution
        self.branch_left = nn.Sequential(
            ConvBNRuLu(i_features, 192, 1),
            ConvBNRuLu(192, 320, 3, 2)
        )

        # 串联 1 × 1 and 7 × 7 and 7 × 7 and 3 × 3 (/2) convolution
        self.branch_middle = nn.Sequential(
            ConvBNRuLu(i_features, 192, 1),
            ConvBNRuLu(192, 192, (1, 7), zero_padding_shape=(0, 3)),
            ConvBNRuLu(192, 192, (7, 1), zero_padding_shape=(3, 0)),
            ConvBNRuLu(192, 192, 3, 2)
        )

        # max pool 3 × 3 (/2)
        self.branch_right = nn.MaxPool2d(3, 2)

    def forward(self, x):
        """前馈函数"""

        # 返回 Batch size × 1280 × 8 × 8
        return torch.cat([self.branch_left(x), self.branch_middle(x), self.branch_right(x)], 1)


class InceptionFig7(nn.Module):
    """Inception modules with expanded the filter bank outputs"""

    def __init__(self, i_features):
        """
        定义四个分支网络结构，初始化各层参数
        :param i_features: base/input features数量
        """

        super().__init__()

        # 1 × 1 convolution 分支
        self.branch_01 = ConvBNRuLu(i_features, 320, 1)

        # series connect 1 × 1 and [parallel connect 1 × 3 and 3 × 1] convolution 分支
        # 无法在__init__中定义串并结构，这里只定义需要用到的单元层，网络结构留给forward()函数定义前馈流程的同时一并完成
        self.branch_02_1 = ConvBNRuLu(i_features, 384, 1)
        self.branch_02_2l = ConvBNRuLu(384, 384, (1, 3), zero_padding_shape=(0, 1))
        self.branch_02_2r = ConvBNRuLu(384, 384, (3, 1), zero_padding_shape=(1, 0))

        # series connect 1 × 1 and 3 × 3 and [parallel connect 1 × 3 and 3 × 1] convolution 分支
        # 同样无法在__init__中定义串并结构，这里只定义需要用到的单元层，网络结构留给forward()函数定义前馈流程的同时一并完成
        self.branch_03_1 = ConvBNRuLu(i_features, 448, 1)
        self.branch_03_2 = ConvBNRuLu(448, 384, 3, zero_padding_shape=1)
        self.branch_03_3l = ConvBNRuLu(384, 384, (1, 3), zero_padding_shape=(0, 1))
        self.branch_03_3r = ConvBNRuLu(384, 384, (3, 1), zero_padding_shape=(1, 0))

        # average pooling 分支
        self.branch_04 = nn.Sequential(
            nn.AvgPool2d(3, 1, 1),
            ConvBNRuLu(i_features, 192, 1)
        )

    def forward(self, x):
        """前馈函数"""

        # 定义两层串并结构的前馈路径，体现串并结构
        branch_2_1 = self.branch_02_1(x)
        branch_2_2 = torch.cat([self.branch_02_2l(branch_2_1), self.branch_02_2r(branch_2_1)], 1)

        # 定义三层串并结构的前馈路径，体现串并结构
        branch_3_1 = self.branch_03_1(x)
        branch_3_2 = self.branch_03_2(branch_3_1)
        branch_3_3 = torch.cat([self.branch_03_3l(branch_3_2), self.branch_03_3r(branch_3_2)], 1)

        # 返回 Batch size × 2048 × 8 × 8
        return torch.cat([self.branch_01(x), branch_2_2, branch_3_3, self.branch_04(x)], 1)


class MyInceptionV3(nn.Module):
    """Neural Networks of Inception V3"""

    def __init__(self, num_classes: int = 1000, dropout: float = 0.5):
        """
        定义网络结构，初始化参数
        :param num_classes: 分类器最终神经元个数，即分类任务涉及类别数
        :param dropout: dropout概率
        """

        super().__init__()

        self.whole_net = nn.Sequential(
            # input: B × 3 × 299 × 299
            ConvBNRuLu(3, 32, 3, 2),
            # 32 × 149 × 149
            ConvBNRuLu(32, 32, 1),
            # 32 × 147 × 147
            ConvBNRuLu(32, 64, 3, zero_padding_shape=1),
            # 64 × 147 × 147
            nn.MaxPool2d(3, 2),
            # 64 × 73 × 73
            ConvBNRuLu(64, 80, 3),
            # 80 × 71 × 71
            ConvBNRuLu(80, 192, 3, 2),
            # 192 × 35 × 35
            ConvBNRuLu(192, 192, 3, zero_padding_shape=1),
            # 192 × 35 × 35
            InceptionFig5(192, pool_features=32),
            # 256 × 35 × 35
            InceptionFig5(256, pool_features=64),
            # 288 × 35 × 35
            InceptionFig5(288, pool_features=64),
            # 288 × 35 × 35
            InceptionFig5ToFig6(288),
            # 768 × 17 × 17
            InceptionFig6(768, channels_inside=128),
            # 768 × 17 × 17
            InceptionFig6(768, channels_inside=160),
            # 768 × 17 × 17
            InceptionFig6(768, channels_inside=160),
            # 768 × 17 × 17
            InceptionFig6(768, channels_inside=192),
            # 768 × 17 × 17
            InceptionFig6ToFig7(768),
            # 1280 × 8 × 8
            InceptionFig7(1280),
            # 2048 × 8 × 8
            InceptionFig7(2048),
            # 2048 × 8 × 8
            nn.AdaptiveAvgPool2d(1)
        )

        self.drop_out = nn.Dropout(p=dropout)

        self.fc1 = nn.Linear(2048, num_classes)

    def forward(self, x):
        """前馈函数"""

        # 完成特征提取网络部分
        x = self.whole_net(x)
        # 添加dropout
        x = self.drop_out(x)
        # 平铺
        x = torch.flatten(x, 1)
        # 分类网络进行分类
        x = self.fc1(x)

        return x
