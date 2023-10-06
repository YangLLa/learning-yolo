# ！/user/bin/env python
# -*- coding:UTF-8 -*-
# author: yanglulu time:2023/3/10

import torch
from torch import nn
import math


class Conv_Norm_Acti(nn.Module):
    def __init__ (self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(Conv_Norm_Acti, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                              kernel_size=kernel_size, stride=stride, padding=padding)
        self.norm = nn.BatchNorm2d(num_features=out_channels)
        self.acti = nn.ReLU(inplace=True)

    def forward (self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.acti(x)
        return x


class Stem_Block(nn.Module):
    """
    根模块
    """

    def __init__ (self, inp_channel=3, out_channels=32):
        super(Stem_Block, self).__init__()
        half_out_channels = int(out_channels / 2)
        self.conv_3x3_1 = Conv_Norm_Acti(in_channels=inp_channel, out_channels=out_channels,
                                         kernel_size=3, stride=2, padding=1)
        self.conv_3x3_2 = Conv_Norm_Acti(in_channels=16, out_channels=out_channels,
                                         kernel_size=3, stride=2, padding=1)
        self.conv_1x1_1 = Conv_Norm_Acti(in_channels=32, out_channels=half_out_channels, kernel_size=1)
        self.conv_1x1_2 = Conv_Norm_Acti(in_channels=64, out_channels=out_channels, kernel_size=1)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward (self, x):
        x = self.conv_3x3_1(x)

        x1 = self.conv_1x1_1(x)
        x1 = self.conv_3x3_2(x1)
        x2 = self.max_pool(x)
        x_cat = torch.cat((x1, x2), dim=1)
        x_out = self.conv_1x1_2(x_cat)
        return x_out


class Two_way_dense_layer(nn.Module):
    """
    特征提取的主力
    """
    base_channel_num = 32

    def __init__ (self, inp_channel, bottleneck_wid, growthrate):
        super(Two_way_dense_layer, self).__init__()
        growth_channel = self.base_channel_num * growthrate
        growth_channel = int(growth_channel / 2)
        bottleneck_out = int(growth_channel * bottleneck_wid / 4)

        if bottleneck_out > inp_channel / 2:
            bottleneck_out = int(bottleneck_out / 8) * 4
            print("bottleneck_out is too big,adjust it to:", bottleneck_out)

        self.conv_1x1 = Conv_Norm_Acti(in_channels=inp_channel, out_channels=bottleneck_out,
                                       kernel_size=1)
        self.conv_3x3_1 = Conv_Norm_Acti(in_channels=bottleneck_out, out_channels=growth_channel,
                                         kernel_size=3, padding=1)
        self.conv_3x3_2 = Conv_Norm_Acti(in_channels=growth_channel, out_channels=growth_channel,
                                         kernel_size=3, padding=1)

    def forward (self, x):
        x_branch = self.conv_1x1(x)
        x_branch_1 = self.conv_3x3_1(x_branch)
        x_branch_2 = self.conv_3x3_1(x_branch)
        x_branch_2 = self.conv_3x3_2(x_branch_2)
        out = torch.cat((x, x_branch_1, x_branch_2), dim=1)
        return out


class Dense_Block(nn.Module):
    def __init__(self,layer_num,inp_channel,bottleneck_wid,growthrate):
        super(Dense_Block,self).__init__()
        self.layers = nn.Sequential()
        base_channel_num = Two_way_dense_layer.base_channel_num
        for i in range(layer_num):
            layer = Two_way_dense_layer(inp_channel+i*growthrate*base_channel_num,
                                        bottleneck_wid,growthrate)
            self.layers.add_module("denselayer%d"%(i+1),layer)
    def forward(self,x):
        x = self.layers(x)
        return x

class Transition_layer(nn.Module):
    def __init__(self,inp_channel,use_pool=True):
        super(Transition_layer,self).__init__()
        self.conv_1x1 = Conv_Norm_Acti(in_channels=inp_channel,out_channels=inp_channel,kernel_size=1)
        self.avg_pool = nn.AvgPool2d(kernel_size=2,stride=2)
        self.use_pool = use_pool
    def forward(self,x):
        x = self.conv_1x1(x)
        if self.use_pool:
            x = self.avg_pool(x)
        return x


class Peleenet(nn.Module):
    def __init__ (self, growthrate=1, layer_num_cfg=[3, 4, 8, 6], bottleneck_width=[1, 2, 4, 4],
                  inp_channels=[32, 128, 256, 512]):
        super(Peleenet, self).__init__()
        base_channel_num = Two_way_dense_layer.base_channel_num
        self.features = nn.Sequential()
        self.stem_block = Stem_Block()  # stride = 4
        self.features.add_module("Stage_0", self.stem_block)
        assert len(layer_num_cfg) == 4 and len(bottleneck_width), "layer_num_cfg or bottleneck_width 的元素长度不是4!"
        for i in range(4):
            self.stage = Dense_Block(layer_num=layer_num_cfg[i], inp_channel=inp_channels[i],
                                     bottleneck_wid=bottleneck_width[i], growthrate=growthrate)
            if i < 3:
                self.translayer = Transition_layer(
                    inp_channel=inp_channels[i] + base_channel_num * growthrate * layer_num_cfg[i])
            else:
                self.translayer = Transition_layer(
                    inp_channel=inp_channels[i] + base_channel_num * growthrate * layer_num_cfg[i],
                    use_pool=False)
            self.features.add_module("Stage_%d" % (i + 1), self.stage)
            self.features.add_module("Translayer_%d" % (i + 1), self.translayer)
        self._initialize_weights()

    def _initialize_weights (self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward (self, x):
        x = self.features(x)
        return x



if __name__ == "__main__":
    inp = torch.randn((2, 3, 224, 224))
    model = Peleenet(growthrate=1)
    result = model(inp)
    print(result.size())

# 输出结果
"""
torch.Size([2, 704, 7, 7])
"""