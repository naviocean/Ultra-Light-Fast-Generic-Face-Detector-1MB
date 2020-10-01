import torch.nn as nn
import torch.nn.functional as F
from .condconv import CondConv2D
import functools
from vision.utils.common import DWConv, Conv
import math
import torch


class Mb_Tiny(nn.Module):

    def __init__(self, num_classes=2):
        super(Mb_Tiny, self).__init__()
        self.base_channel = 8 * 2
        # Conv2d = functools.partial(CondConv2D, num_experts=2)
        # print('num_experts', 2)

        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                Conv(inp, oup, k=3, s=stride, p=1),
            )

        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                Conv(inp, inp, k=3, s=stride, p=1, g=inp),
                Conv(inp, oup, k=1, s=1, p=0, g=1),
            )

        self.model = nn.Sequential(
            conv_bn(3, self.base_channel, 2),  # 160*120
            conv_dw(self.base_channel, self.base_channel * 2, 1),
            conv_dw(self.base_channel * 2, self.base_channel * 2, 2),  # 80*60
            conv_dw(self.base_channel * 2, self.base_channel * 2, 1),
            conv_dw(self.base_channel * 2, self.base_channel * 4, 2),  # 40*30
            conv_dw(self.base_channel * 4, self.base_channel * 4, 1),
            conv_dw(self.base_channel * 4, self.base_channel * 4, 1),
            conv_dw(self.base_channel * 4, self.base_channel * 4, 1),
            conv_dw(self.base_channel * 4, self.base_channel * 8, 2),  # 20*15
            conv_dw(self.base_channel * 8, self.base_channel * 8, 1),
            conv_dw(self.base_channel * 8, self.base_channel * 8, 1),
            conv_dw(self.base_channel * 8, self.base_channel * 16, 2),  # 10*8
            conv_dw(self.base_channel * 16, self.base_channel * 16, 1),
        )
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.model(x)
        x = F.avg_pool2d(x, 7)
        x = x.view(-1, 1024)
        x = self.fc(x)
        return x
