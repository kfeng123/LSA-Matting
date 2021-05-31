# these codes are adapted from https://github.com/CSAILVison/semantic-segmentation-pytorch/mit_semseg/models/resnet.py

import torch
import torch.nn as nn
import math
#from mit_semseg.lib.nn import SynchronizedBatchNorm2d
#BatchNorm2d = SynchronizedBatchNorm2d
BatchNorm2d = nn.BatchNorm2d

__all__ = ['ResNet', 'resnet18', 'resnet50', 'resnet101'] # resnet101 is coming soon!

#model_urls = {
#    'resnet18': 'http://sceneparsing.csail.mit.edu/model/pretrained_resnet/resnet18-imagenet.pth',
#    'resnet50': 'http://sceneparsing.csail.mit.edu/model/pretrained_resnet/resnet50-imagenet.pth',
#    'resnet101': 'http://sceneparsing.csail.mit.edu/model/pretrained_resnet/resnet101-imagenet.pth'
#}

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, withReLU = True):
        super(BasicBlock, self).__init__()

        self.withReLU = withReLU
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        if self.withReLU:
            out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, withReLU = True):
        super(Bottleneck, self).__init__()
        self.withReLU = withReLU
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        if self.withReLU:
            out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, inChannel = 4):
        self.inplanes = 128
        super(ResNet, self).__init__()
        self.conv1 = conv3x3(inChannel, 64, stride=2)
        self.bn1 = BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(64, 64)
        self.bn2 = BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = conv3x3(64, 128)
        self.bn3 = BatchNorm2d(128)
        self.relu3 = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            if i!= blocks-1:
                layers.append(block(self.inplanes, planes))
            else:
                layers.append(block(self.inplanes, planes, withReLU = False))

        return nn.Sequential(*layers)

    def forward(self, x):

        outputs = {}

        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        outputs['stage1'] = x

        x = self.relu3(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        outputs['stage2'] = x

        x = self.relu3(x)
        x = self.layer2(x)
        outputs['stage3'] = x

        x = self.relu3(x)
        x = self.layer3(x)
        outputs['stage4'] = x

        x = self.relu3(x)
        x = self.layer4(x)
        outputs['stage5'] = x

        return outputs


model_dirs = {
    'resnet18': './pretrained/resnet18-imagenet.pth',
    'resnet50': './pretrained/resnet50-imagenet.pth',
}

def resnet18(pretrained=True, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    model._out_feature_channels = {
            "stage1": 128,
            "stage2": 64,
            "stage3": 128,
            "stage4": 256,
            "stage5": 512,
            }
    if pretrained:
        pretrained_dict = torch.load(model_dirs['resnet18'])
        model_dict = model.state_dict()
        for name in pretrained_dict:
            if name not in model_dict:
                print(name, " not in model_dict!!!!!!!!!")
                continue
            if name == "conv1.weight":
                model_dict[name][:, :3, :, :] = pretrained_dict[name][:,:,:,:]
                model_dict[name][:, 3:, :, :] = 0
                continue
            model_dict[name] = pretrained_dict[name]

        model.load_state_dict(model_dict)

    return model


def resnet50(pretrained=True, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    model._out_feature_channels = {
            "stage1": 128,
            "stage2": 256,
            "stage3": 512,
            "stage4": 1024,
            "stage5": 2048,
            }
    if pretrained:
        pretrained_dict = torch.load(model_dirs['resnet50'])
        model_dict = model.state_dict()
        for name in pretrained_dict:
            if name not in model_dict:
                print(name, " not in model_dict!!!!!!!!!")
                continue
            if name == "conv1.weight":
                model_dict[name][:, :3, :, :] = pretrained_dict[name][:,:,:,:]
                model_dict[name][:, 3:, :, :] = 0
                continue
            model_dict[name] = pretrained_dict[name]

        model.load_state_dict(model_dict)

    return model

