import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import pdb

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)



class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, withReLU = True):
        super(BasicBlock, self).__init__()

        self.withReLU = withReLU

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        if self.withReLU:
            out = self.relu(out)

        return out

class ResNet(nn.Module):

    def __init__(self, block, layers, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(4, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer, withReLU = True))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            if i!= blocks-1:
                layers.append(block(self.inplanes, planes, groups=self.groups,
                                    base_width=self.base_width, dilation=self.dilation,
                                    norm_layer=norm_layer, withReLU = True))
            else:
                layers.append(block(self.inplanes, planes, groups=self.groups,
                                    base_width=self.base_width, dilation=self.dilation,
                                    norm_layer=norm_layer, withReLU = False))

        return nn.Sequential(*layers)

    def _forward(self, x):
        x1 = self.conv1(x)
        x1 = self.bn1(x1)

        x2 = self.relu(x1)
        x2 = self.maxpool(x2)
        x2 = self.layer1(x2)

        x3 = self.relu(x2)
        x3 = self.layer2(x3)

        x4 = self.relu(x3)
        x4 = self.layer3(x4)

        x5 = self.relu(x4)
        x5 = self.layer4(x5)

        return x5

    # Allow for accessing forward method in a inherited class
    forward = _forward

def weight_init(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.kaiming_normal_(m.weight.data)
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()

def resnet18(pretrained="./pretrained/resnet18-5c106cde.pth", **kwargs):
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    model.apply(weight_init)

    if pretrained:
        print("Loading pretrained model!!!!")
        model_dict = model.state_dict()
        pretrained_dict = torch.load(pretrained)

        for name in pretrained_dict:
            if name not in model_dict:
                continue
            if name == "conv1.weight":
                model_weight = model_dict[name]
                assert model_weight.shape[1] == 4
                model_weight[:, 0:3, :, :] = pretrained_dict[name]
                model_weight[:, 3, :, :] = torch.tensor(0)
                model_dict[name] = model_weight
            else:
                model_dict[name] = pretrained_dict[name]
        #model.load_state_dict(model_dict)
    return model

def resnet34(pretrained="./pretrained/resnet34-333f7ec4.pth", **kwargs):
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    model.apply(weight_init)

    if pretrained:
        print("Loading pretrained model!!!!")
        model_dict = model.state_dict()
        pretrained_dict = torch.load(pretrained)

        for name in pretrained_dict:
            if name not in model_dict:
                continue
            if name == "conv1.weight":
                model_weight = model_dict[name]
                assert model_weight.shape[1] == 4
                model_weight[:, 0:3, :, :] = pretrained_dict[name]
                model_weight[:, 3, :, :] = torch.tensor(0)
                model_dict[name] = model_weight
            else:
                model_dict[name] = pretrained_dict[name]
        #model.load_state_dict(model_dict)
    return model



class decoder_module(nn.Module):

    def __init__(self, leftp, inp, oup):
        super(decoder_module, self).__init__()

        self.skip_add = nn.Sequential(
                nn.Conv2d(leftp, inp, 3, 1, 1, bias = False),
                nn.BatchNorm2d(inp),
                #nn.ReLU(inplace=True),
                #nn.Conv2d(inp, inp, 3, 1, 1, bias = False),
                #nn.BatchNorm2d(inp),
                )

        self.skip_mul = nn.Sequential(
                nn.Conv2d(leftp, inp, 3, 1, 1, bias = False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),
                nn.Conv2d(inp, inp, 3, 1, 1, bias = False),
                )

        self.decoder = nn.Sequential(
                nn.ReLU(inplace=True),
                nn.Conv2d(inp, oup, 3, 1, 1, bias = False),
                nn.BatchNorm2d(oup),
                )

    def forward(self, left, lower):
        #theCombined = self.skip_add(left) + torch.sigmoid(self.skip_mul(left)) * lower
        theCombined = self.skip_add(left) + lower
        return self.decoder(theCombined)


class pre_module(nn.Module):
    def __init__(self):
        super(pre_module, self).__init__()
        #self.preConv1 = nn.Sequential(
        #        nn.Conv2d(4, 16, 1, 1, 0),
        #        nn.ReLU(inplace = True),
        #        nn.Conv2d(16, 16, 1, 1, 0),
        #        )
        #self.preConv2 = nn.Conv2d(4, 16, 1, 1, 0, bias = False)
        self.relu = nn.ReLU()

        self.kernel = torch.ones(1,1,3,3).cuda() / 9

    def forward(self, x):
        #return  torch.sigmoid(self.preConv1(x)) * self.preConv2(x)

        #trimap = x[:, 3,:,:]
        #trimap = trimap[:, None, :, :]
        #fore = self.relu(trimap-3./4.) * 4
        #back = self.relu(1./4. - trimap) * 4
        #tri = (trimap -fore) * 2
        #return torch.cat([x[:,:3,:,:], x[:,:3,:,:] * fore, x[:,:3,:,:] * back, x[:,:3,:,:] * tri ], 1)

        trimap = x[:, 3,:,:]
        trimap = trimap[:, None, :, :]
        inner0 = self.relu(trimap-3./4.) * 4
        innerList = [inner0]
        for i in range(5):
            innerList.append(
                        F.conv2d(innerList[-1], self.kernel, padding = 1)
                    )
        outter0 = 1 - self.relu(1./4. - trimap) * 4
        outterList = [outter0]
        for i in range(5):
            outterList.append(
                        F.conv2d(outterList[-1], self.kernel, padding = 1)
                    )
        return torch.cat([x[:,:3,:,:]] + innerList + outterList, 1)



class theModel(nn.Module):

    def __init__(self, backbone):
        super(theModel, self).__init__()
        if backbone == "res18":
            resnet = resnet18()
        elif backbone == "res34":
            resnet = resnet34()


        self.pre_module = pre_module()

        self.conv1=conv3x3(15, 64, 2)
        self.bn1=resnet.bn1
        self.relu=resnet.relu

        #self.conv1=resnet.conv1
        #self.bn1=resnet.bn1
        #self.relu=resnet.relu

        # add two convs at stride == 2
        #self.add_conv2 = nn.Sequential(
        #        conv3x3(64, 64),
        #        nn.BatchNorm2d(64),
        #        self.relu,
        #        conv3x3(64, 64),
        #        nn.BatchNorm2d(64),
        #        )
        self.add_conv2 = nn.Sequential(
                #BasicBlock(64, 64, withReLU = True),
                BasicBlock(64, 64, withReLU = False),
                )


        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4


        decoder_channels_4 = 64
        decoder_channels_3 = 64
        decoder_channels_2 = 64
        decoder_channels_1 = 64
        decoder_channels_0 = 32

        self.decoder_5 = nn.Sequential(
                nn.ReLU(inplace=True),
                nn.Conv2d(512, decoder_channels_4, 3, 1, 1, bias = False),
                nn.BatchNorm2d(decoder_channels_4),
                nn.ReLU(inplace=True),
                nn.Conv2d(decoder_channels_4, decoder_channels_4, 3, 1, 1, bias = False),
                nn.BatchNorm2d(decoder_channels_4),
                )

        self.decoder_4 = decoder_module(256, decoder_channels_4, decoder_channels_3)
        self.decoder_3 = decoder_module(128, decoder_channels_3, decoder_channels_2)
        self.decoder_2 = decoder_module(64, decoder_channels_2, decoder_channels_1)
        self.decoder_1 = decoder_module(64, decoder_channels_1, decoder_channels_0)

        self.decoder_0 = nn.Sequential(
                nn.ReLU(inplace=True),
                nn.Conv2d(decoder_channels_0, decoder_channels_0, 3, 1, 1, bias = False),
                nn.BatchNorm2d(decoder_channels_0),
                nn.ReLU(inplace=True),
                nn.Conv2d(decoder_channels_0, 1, 1, 1, 0),
                )
        self.skip_0 = nn.Sequential(
                nn.Conv2d(4, decoder_channels_0, 3, 1, 1, bias = False),
                nn.BatchNorm2d(decoder_channels_0),
                nn.ReLU(inplace=True),
                nn.Conv2d(decoder_channels_0, decoder_channels_0, 3, 1, 1, bias = False),
                nn.BatchNorm2d(decoder_channels_0),
                )

    def forward(self, x):
        #x1 = self.conv1(x)
        #x1 = self.bn1(x1)
        #x1 = self.relu(x1)

        x0 = self.pre_module(x)

        x1 = self.conv1(x0)
        x1 = self.bn1(x1)
        x1 = self.relu(x1)
        # newly added
        x1 = self.add_conv2(x1)

        x2 = self.relu(x1)
        x2 = self.maxpool(x2)
        x2 = self.layer1(x2)

        x3 = self.relu(x2)
        x3 = self.layer2(x3)

        x4 = self.relu(x3)
        x4 = self.layer3(x4)

        x5 = self.relu(x4)
        x5 = self.layer4(x5)


        #x5 = self.pam(x5)

        x5_decoder = self.decoder_5(x5)
        x4_decoder = F.interpolate(x5_decoder, x4.shape[2:], mode = "bilinear")
        #x4_decoder = self.decoder_4(torch.cat((x4_decoder, x4), dim = 1))
        x4_decoder = self.decoder_4(x4, x4_decoder)

        x3_decoder = F.interpolate(x4_decoder, x3.shape[2:], mode = "bilinear")
        x3_decoder = self.decoder_3(x3, x3_decoder)

        x2_decoder = F.interpolate(x3_decoder, x2.shape[2:], mode = "bilinear")
        x2_decoder = self.decoder_2(x2, x2_decoder)

        x1_decoder = F.interpolate(x2_decoder, x1.shape[2:], mode = "bilinear")
        x1_decoder = self.decoder_1(x1, x1_decoder)

        x0_decoder = F.interpolate(x1_decoder, x.shape[2:], mode = "bilinear")
        #pred_alpha = self.decoder_0(x0_decoder)
        pred_alpha = self.decoder_0(self.skip_0(x) + x0_decoder)
        #pred_alpha = torch.sigmoid(pred_alpha)

        return pred_alpha




if __name__ == "__main__":

    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '6'
    model = resnet34()
    model.eval()
    model.cuda()
    dump_x = torch.randn(1, 4, 2048, 2048).cuda()
    y = model(dump_x)
    pdb.set_trace()


