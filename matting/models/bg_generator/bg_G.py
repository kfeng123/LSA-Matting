from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual

        return out

class bg_G(nn.Module):
    def __init__(self):
        super(bg_G, self).__init__()
        self.conv1 = nn.Conv2d(6, 8, 1, 1, 0, bias = False)
        self.block1 = BasicBlock(8,8)
        self.block2 = BasicBlock(8,8)
        self.conv2 = nn.Conv2d(8, 3, 1, 1, 0, bias = False)
        self.bn = nn.BatchNorm2d(3, affine = False)

    def forward(self, x, noise):
        ori = x
        x = torch.cat((x, F.interpolate(noise, x.shape[2:], mode = "bilinear")), 1)
        x = self.conv1(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.conv2(x)
        x = self.bn(x)
        return x

