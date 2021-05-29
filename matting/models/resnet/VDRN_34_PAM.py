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
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
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
        out = self.relu(out)

        return out



class Bottleneck(nn.Module):
    expansion = 4
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class Refinement(nn.Module):
    def __init__(self):
        super(Refinement, self).__init__()
        self.branch1 = nn.Sequential(
                nn.Conv2d(4, 16, 5, 1, 2, bias = False),
                nn.BatchNorm2d(16),
                nn.Conv2d(16, 16, 5, 1, 2, bias = False),
                nn.BatchNorm2d(16),
                nn.Conv2d(16, 1, 1, 1, 0),
                )

    def forward(self, x, alpha):
        return torch.sigmoid(self.branch1(torch.cat((x, alpha), 1)))


class SR_Res_Block(nn.Module):
    def __init__(self, inplanes):
        super(SR_Res_Block, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, inplanes //4, 3, 1, 1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(inplanes //4, inplanes, 3, 1, 1)
    def forward(self, x):
            identity = x
            out = self.conv1(x)
            out = self.relu(out)
            out = self.conv2(out)
            out += identity
            return out

class Refinement_good(nn.Module):
    def __init__(self):
        super(Refinement_good, self).__init__()
        self.branch1 = nn.Sequential(
                nn.Conv2d(4, 16, 5, 1, 2, bias = False),
                nn.Conv2d(16, 16, 5, 1, 2, bias = False),
                nn.Conv2d(16, 1, 1, 1, 0),
                )
        self.branch2 = nn.Sequential(
                nn.Conv2d(4, 16, 3, 1, 1, bias = False),
                SR_Res_Block(16), SR_Res_Block(16), SR_Res_Block(16), SR_Res_Block(16),
                SR_Res_Block(16), SR_Res_Block(16), SR_Res_Block(16), SR_Res_Block(16),
                SR_Res_Block(16), SR_Res_Block(16), SR_Res_Block(16), SR_Res_Block(16),
                SR_Res_Block(16), SR_Res_Block(16), SR_Res_Block(16), SR_Res_Block(16),
                nn.Conv2d(16, 1, 3, 1, 1),
                )

    def forward(self, x, alpha):
        theInput = torch.cat((x, alpha), 1)
        return torch.sigmoid(self.branch1(theInput) + self.branch2(theInput))




class PAM_Module(nn.Module):
    """ Position attention module"""
    #Ref from SAGAN
    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)
    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
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

        self.decoder_5 = nn.Sequential(
                nn.Conv2d(512, 256, 3, 1, 1, bias = False),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.Conv2d(256, 256, 3, 1, 1, bias = False),
                nn.BatchNorm2d(256),
                nn.ReLU()
                )
        self.decoder_4 = nn.Sequential(
                nn.Conv2d(512, 128, 3, 1, 1, bias = False),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.Conv2d(128, 128, 3, 1, 1, bias = False),
                nn.BatchNorm2d(128),
                nn.ReLU()
                )
        self.decoder_3 = nn.Sequential(
                nn.Conv2d(256, 64, 3, 1, 1, bias = False),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64, 64, 3, 1, 1, bias = False),
                nn.BatchNorm2d(64),
                nn.ReLU()
                )
        self.decoder_2 = nn.Sequential(
                nn.Conv2d(128, 64, 3, 1, 1, bias = False),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64, 64, 3, 1, 1, bias = False),
                nn.BatchNorm2d(64),
                nn.ReLU()
                )
        self.decoder_1 = nn.Sequential(
                nn.Conv2d(128, 64, 3, 1, 1, bias = False),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64, 64, 3, 1, 1, bias = False),
                nn.BatchNorm2d(64),
                nn.ReLU()
                )
        self.decoder_0 = nn.Sequential(
                nn.Conv2d(64, 1, 1, 1, 0),
                )

        self.pam = PAM_Module(512)

        #self.Refinement_good = Refinement_good()


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
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward(self, x):
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = self.relu(x1)

        x2 = self.maxpool(x1)
        x2 = self.layer1(x2)

        x3 = self.layer2(x2)

        x4 = self.layer3(x3)

        x5 = self.layer4(x4)

        x5 = self.pam(x5)

        x5_decoder = self.decoder_5(x5)

        x4_decoder = F.interpolate(x5_decoder, x4.shape[2:], mode = "bilinear")
        x4_decoder = self.decoder_4(torch.cat((x4_decoder, x4), dim = 1))

        x3_decoder = F.interpolate(x4_decoder, x3.shape[2:], mode = "bilinear")
        x3_decoder = self.decoder_3(torch.cat((x3_decoder, x3), dim = 1))

        x2_decoder = F.interpolate(x3_decoder, x2.shape[2:], mode = "bilinear")
        x2_decoder = self.decoder_2(torch.cat((x2_decoder, x2), dim = 1))

        x1_decoder = F.interpolate(x2_decoder, x1.shape[2:], mode = "bilinear")
        x1_decoder = self.decoder_1(torch.cat((x1_decoder, x1), dim = 1))

        x0_decoder = F.interpolate(x1_decoder, x.shape[2:], mode = "bilinear")
        pred_alpha = self.decoder_0(x0_decoder)
        pred_alpha = torch.sigmoid(pred_alpha)
        #pred_alpha_refined = self.Refinement_good(x[:,:3,:,:], pred_alpha)
        return pred_alpha

    # Allow for accessing forward method in a inherited class
    forward = _forward



def weight_init(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.kaiming_normal_(m.weight.data)
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()

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




if __name__ == "__main__":

    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    model = resnet34()
    model.eval()
    model.cuda()
    dump_x = torch.randn(1, 4, 2048, 2048).cuda()
    y = model(dump_x)
    pdb.set_trace()


