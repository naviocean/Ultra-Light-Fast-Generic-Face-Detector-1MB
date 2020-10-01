# This file contains modules common to various models
import math

import torch
import torch.nn as nn


def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


def DWConv(c1, c2, k=1, s=1, p=None, d=1, g=None, act=True):
    # Depthwise convolution
    if g is None:
        return Conv(c1, c2, k, s, p, d=d, g=math.gcd(c1, c2), act=act)
    return Conv(c1, c2, k, s, p, d=d, g=g,  act=act)


class Conv(nn.Module):
    # Standard convolution
    # ch_in, ch_out, kernel, stride, padding, groups
    def __init__(self, c1, c2, k=1, s=1, p=None, d=1, g=1, b=False, act=True, bn=True, relu=True):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), dilation=d,
                              groups=g, bias=b)
        self.bn = nn.BatchNorm2d(c2) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None
        self.act = nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        x = self.act(x)
        return x

    def fuseforward(self, x):
        x = self.conv(x)
        if self.relu is not None:
            x = self.relu(x)
        x = self.act(x)
        return x


class BasicRFB(nn.Module):

    def __init__(self, in_planes, out_planes, stride=1, scale=0.1, map_reduce=8, vision=1, groups=1):
        super(BasicRFB, self).__init__()
        self.scale = scale
        self.out_channels = out_planes
        inter_planes = in_planes // map_reduce

        self.branch0 = nn.Sequential(
            Conv(in_planes, inter_planes, k=1, s=1,
                 p=0, g=groups, relu=True),
            Conv(inter_planes, 2 * inter_planes, k=(3, 3),
                 s=stride, p=(1, 1), g=groups),
            Conv(2 * inter_planes, 2 * inter_planes, k=3, s=1,
                 p=vision + 1, d=vision + 1, g=groups, relu=True)
        )

        self.branch1 = nn.Sequential(
            Conv(in_planes, inter_planes, k=1, s=1,
                 p=0, g=groups, relu=True),
            Conv(inter_planes, 2 * inter_planes, k=(3, 3), s=stride,
                 p=(1, 1), g=groups),
            Conv(2 * inter_planes, 2 * inter_planes, k=3, s=1,
                 p=vision + 2, d=vision + 2, g=groups, relu=True),
        )

        self.branch2 = nn.Sequential(
            Conv(in_planes, inter_planes, k=1, s=1,
                 p=0, g=groups, relu=True),
            Conv(inter_planes, (inter_planes // 2) * 3, k=3, s=1,
                 p=1, g=groups),
            Conv((inter_planes // 2) * 3, 2 * inter_planes, k=3, s=stride,
                 p=1, g=groups),
            Conv(2 * inter_planes, 2 * inter_planes, k=3, s=1,
                 p=vision + 4, d=vision + 4,  g=groups, relu=True),
        )
        self.ConvLinear = Conv(
            6 * inter_planes, out_planes, k=1, s=1, p=0, relu=True)
        self.shortcut = Conv(
            in_planes, out_planes, k=1, s=stride, p=0, relu=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)

        out = torch.cat((x0, x1, x2), 1)
        out = self.ConvLinear(out)
        short = self.shortcut(x)
        out = out * self.scale + short
        out = self.relu(out)

        return out
