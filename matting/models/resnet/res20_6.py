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
        print("Loading pretrained backbone model resnet18!!!!")
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
        model.load_state_dict(model_dict)
    return model

def resnet34(pretrained="./pretrained/resnet34-333f7ec4.pth", **kwargs):
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    model.apply(weight_init)

    if pretrained:
        print("Loading pretrained backbone model resnet34!!!!")
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
        model.load_state_dict(model_dict)
    return model



class decoder_module(nn.Module):

    def __init__(self):
        super(decoder_module, self).__init__()

        self.to_low = nn.Sequential(
                nn.PReLU(64),
                nn.Conv2d(64, 64, 3, 1, 1, bias = False),
                nn.BatchNorm2d(64),
                )

        self.to_high = nn.Sequential(
                nn.PReLU(64),
                nn.Conv2d(64, 64, 3, 1, 1, bias = False),
                nn.BatchNorm2d(64),
                )

        self.down = nn.AvgPool2d(kernel_size = (2, 2), stride = 2)
        self.up = nn.Upsample(scale_factor = 2, mode = "nearest")


    def forward(self, left_high, left_low, high, low):
        high_out = self.to_high(self.up(left_low) + high)
        low_out = self.to_low(self.down(left_high) + low)

        return self.up(high_out), self.up(low_out)


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


class my_attention(nn.Module):
    def __init__(self, x_dim, ref_dim):
        super(my_attention, self).__init__()

        mid_channels = 16

        self.conv_attention_x = nn.Conv2d(in_channels=x_dim, out_channels=mid_channels, kernel_size=1)
        self.conv_attention_ref = nn.Conv2d(in_channels=ref_dim, out_channels=mid_channels, kernel_size=1)

        self.conv_feature_ref = nn.Conv2d(in_channels=ref_dim, out_channels=mid_channels, kernel_size=1)

        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=1)
    def forward(self, x, ref):

        m_batchsize, C, height, width = x.size()
        attention_x = self.conv_attention_x(x).view(m_batchsize, -1, width*height)
        _, _, height_ref, width_ref = ref.size()
        attention_ref = self.conv_attention_ref(ref).view(m_batchsize, -1, width_ref*height_ref).permute(0, 2, 1)
        attention = torch.bmm(attention_ref, attention_x)
        attention = self.softmax(attention)

        feature_ref = self.conv_feature_ref(ref).view(m_batchsize, -1, width_ref * height_ref)

        out = torch.bmm(feature_ref, attention)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out




class my_local_attention(nn.Module):
    def __init__(self, x_dim):
        super(my_local_attention, self).__init__()

        self.kernel_size = 3

        self.padding = nn.ConstantPad2d(self.kernel_size//2, 0)

        mid_channels = 32
        self.mid_channels = mid_channels

        self.conv_attention_x1 = nn.Conv2d(in_channels=x_dim, out_channels=mid_channels, kernel_size=1)
        self.conv_attention_x2 = nn.Conv2d(in_channels=x_dim, out_channels=mid_channels, kernel_size=1)

        self.conv_attention_x3 = nn.Conv2d(in_channels=x_dim, out_channels=x_dim, kernel_size=1)
        #self.conv_feature_ref = nn.Conv2d(in_channels=x_dim, out_channels=mid_channels, kernel_size=1)

        #self.the_kernels = []
        #for i in range(self.kernel_size):
        #    for j in range(self.kernel_size):
        #        tmp = torch.zeros((mid_channels, 1, self.kernel_size, self.kernel_size)).cuda()
        #        tmp[:,:,i,j] = 1
        #        self.the_kernels.append(tmp)

        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=1)
    def forward(self, x):
        m_batchsize, C, height, width = x.size()

        x1 = self.conv_attention_x1(x)
        x2 = self.conv_attention_x2(x)
        x3 = self.conv_attention_x3(x)

        x1 = self.padding(x1)

        x3 = self.padding(x3)


        for i in range(self.kernel_size):
            for j in range(self.kernel_size):
                x = x + self.gamma * x3[:,:,i:i+height, j:j+width] * ( torch.sigmoid(x2 * x1[:,:,i:i+height,j:j+width]).sum(dim=1, keepdim = True) )
        return x





class theModel(nn.Module):

    def __init__(self, backbone):
        super(theModel, self).__init__()
        if backbone == "res18":
            resnet = resnet18()
        elif backbone == "res34":
            resnet = resnet34()

        self.conv1=resnet.conv1
        self.bn1=resnet.bn1
        #self.relu=resnet.relu
        self.relu=nn.ReLU()

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

        self.layer5 = nn.Sequential(
                BasicBlock(512, 512, stride = 2,
                    downsample = nn.Sequential(conv1x1(512, 512, 2), nn.BatchNorm2d(512)),
                    withReLU = True),
                BasicBlock(512, 512, stride = 1, withReLU = False),
                )

        decoder_channels_6 = 64
        decoder_channels_5 = 64
        decoder_channels_4 = 64
        decoder_channels_3 = 64
        decoder_channels_2 = 64
        decoder_channels_1 = 64
        decoder_channels_0 = 32


        self.skip_6 = nn.Sequential(
                nn.Conv2d(512, decoder_channels_6, 3, 1, 1, bias = False),
                nn.BatchNorm2d(decoder_channels_6),
                )
        self.skip_5 = nn.Sequential(
                nn.Conv2d(512, decoder_channels_5, 3, 1, 1, bias = False),
                nn.BatchNorm2d(decoder_channels_5),
                )

        self.skip_4 = nn.Sequential(
                nn.Conv2d(256, decoder_channels_4, 3, 1, 1, bias = False),
                nn.BatchNorm2d(decoder_channels_4),
                )
        self.skip_3 = nn.Sequential(
                nn.Conv2d(128, decoder_channels_3, 3, 1, 1, bias = False),
                nn.BatchNorm2d(decoder_channels_3),
                )
        self.skip_2 = nn.Sequential(
                nn.Conv2d(64, decoder_channels_2, 3, 1, 1, bias = False),
                nn.BatchNorm2d(decoder_channels_2),
                )
        self.skip_1 = nn.Sequential(
                nn.Conv2d(64, decoder_channels_1, 3, 1, 1, bias = False),
                nn.BatchNorm2d(decoder_channels_1),
                )
        self.skip_0 = nn.Sequential(
                nn.Conv2d(4, decoder_channels_0, 3, 1, 1, bias = False),
                nn.BatchNorm2d(decoder_channels_0),
                nn.ReLU(decoder_channels_0),
                nn.Conv2d(decoder_channels_0, decoder_channels_0, 3, 1, 1, bias = False),
                nn.BatchNorm2d(decoder_channels_0),
                )

        self.decoder5 = decoder_module()
        self.decoder4 = decoder_module()
        self.decoder3 = decoder_module()
        self.decoder2 = decoder_module()
        self.decoder1 = decoder_module()

        self.decoder_0 = nn.Sequential(
                nn.PReLU(decoder_channels_0),
                nn.Conv2d(decoder_channels_0, decoder_channels_0, 3, 1, 1, bias = False),
                nn.BatchNorm2d(decoder_channels_0),
                nn.PReLU(decoder_channels_0),
                nn.Conv2d(decoder_channels_0, 1, 1, 1, 0),
                )

        self.conv_haha = nn.Conv2d(64, 32, 3, 1, 1, bias = False)

        self.up = nn.Upsample(scale_factor = 2, mode = "nearest")

    def forward(self, x):
        #x1 = self.conv1(x)
        #x1 = self.bn1(x1)
        #x1 = self.relu(x1)

        #x0 = self.pre_module(x)

        skip0 = self.skip_0(x)

        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = self.relu(x1)
        # 64 * 256 * 256

        # newly added
        x1 = self.add_conv2(x1)
        # 64 * 256 * 256
        skip1 = self.skip_1(x1)

        x2 = self.relu(x1)
        x2 = self.maxpool(x2)
        x2 = self.layer1(x2)
        # 64 * 128 * 128
        skip2 = self.skip_2(x2)

        x3 = self.relu(x2)
        x3 = self.layer2(x3)
        # 128 * 64 * 64
        skip3 = self.skip_3(x3)

        x4 = self.relu(x3)
        x4 = self.layer3(x4)
        # 256 * 32 * 32
        skip4 = self.skip_4(x4)

        x5 = self.relu(x4)
        x5 = self.layer4(x5)
        # 512 * 16 * 16
        skip5 = self.skip_5(x5)

        x6 = self.relu(x5)
        x6 = self.layer5(x6)
        # 512 * 16 * 16
        skip6 = self.skip_6(x6)


        x_high = self.up(skip5)
        x_low = self.up(skip6)
        x_high, x_low = self.decoder4(skip4, skip5, x_high, x_low)
        x_high, x_low = self.decoder3(skip3, skip4, x_high, x_low)
        x_high, x_low = self.decoder2(skip2, skip3, x_high, x_low)
        x_high, x_low = self.decoder1(skip1, skip2, x_high, x_low)
        x_high = self.conv_haha(x_high + self.up(x_low)) + skip0

        pred_alpha = self.decoder_0(x_high)

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



