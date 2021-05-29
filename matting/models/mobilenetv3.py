"""
Creates a MobileNetV3 Model as defined in:
Andrew Howard, Mark Sandler, Grace Chu, Liang-Chieh Chen, Bo Chen, Mingxing Tan, Weijun Wang, Yukun Zhu, Ruoming Pang, Vijay Vasudevan, Quoc V. Le, Hartwig Adam. (2019).
Searching for MobileNetV3
arXiv preprint arXiv:1905.02244.
"""
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


__all__ = ['mobilenetv3_large', 'mobilenetv3_small']


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, _make_divisible(channel // reduction, 8)),
                nn.ReLU(inplace=True),
                nn.Linear(_make_divisible(channel // reduction, 8), channel),
                h_sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


def conv_3x3_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        h_swish()
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        h_swish()
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp, hidden_dim, oup, kernel_size, stride, use_se, use_hs):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]

        self.identity = stride == 1 and inp == oup

        if inp == hidden_dim:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, (kernel_size - 1) // 2, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                h_swish() if use_hs else nn.ReLU(inplace=True),
                # Squeeze-and-Excite
                SELayer(hidden_dim) if use_se else nn.Identity(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                h_swish() if use_hs else nn.ReLU(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, (kernel_size - 1) // 2, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                # Squeeze-and-Excite
                SELayer(hidden_dim) if use_se else nn.Identity(),
                h_swish() if use_hs else nn.ReLU(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV3(nn.Module):
    def __init__(self, cfgs, mode, width_mult=1., image_channel = 4):
        super(MobileNetV3, self).__init__()
        # setting of inverted residual blocks
        self.cfgs = cfgs
        assert mode in ['large', 'small']

        # building first layer
        input_channel = _make_divisible(16 * width_mult, 8)
        layers = [conv_3x3_bn(image_channel, input_channel, 2)]
        # building inverted residual blocks
        block = InvertedResidual
        for k, t, c, use_se, use_hs, s in self.cfgs:
            output_channel = _make_divisible(c * width_mult, 8)
            exp_size = _make_divisible(input_channel * t, 8)
            layers.append(block(input_channel, exp_size, output_channel, k, s, use_se, use_hs))
            input_channel = output_channel
        #self.features = nn.Sequential(*layers)
        self.features = nn.ModuleList(layers)

        self._initialize_weights()

    def forward(self, x):
        output = {}
        # stage 1
        x1 = self.features[0](x)
        x1 = self.features[1](x1)
        output['stage1'] = x1

        # stage 2
        x2 = self.features[2](x1)
        x2 = self.features[3](x2)
        output['stage2'] = x2

        # stage 3
        x3 = self.features[4](x2)
        x3 = self.features[5](x3)
        x3 = self.features[6](x3)
        output['stage3'] = x3

        # stage 4
        x4 = self.features[7](x3)
        x4 = self.features[8](x4)
        x4 = self.features[9](x4)
        x4 = self.features[10](x4)
        x4 = self.features[11](x4)
        x4 = self.features[12](x4)
        output['stage4'] = x4

        # stage 5
        x5 = self.features[13](x4)
        x5 = self.features[14](x5)
        x5 = self.features[15](x5)
        output['stage5'] = x5

        return output

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def mobilenetv3_large(**kwargs):
    """
    Constructs a MobileNetV3-Large model
    """
    cfgs = [
        # k,  t,   c,SE,HS, s
        [3,   1,  16, 0, 0, 1],
        [3,   4,  24, 0, 0, 2],
        [3,   3,  24, 0, 0, 1],
        [5,   3,  40, 1, 0, 2],
        [5,   3,  40, 1, 0, 1],
        [5,   3,  40, 1, 0, 1],
        [3,   6,  80, 0, 1, 2],
        [3, 2.5,  80, 0, 1, 1],
        [3, 2.3,  80, 0, 1, 1],
        [3, 2.3,  80, 0, 1, 1],
        [3,   6, 112, 1, 1, 1],
        [3,   6, 112, 1, 1, 1],
        [5,   6, 160, 1, 1, 2],
        [5,   6, 160, 1, 1, 1],
        [5,   6, 160, 1, 1, 1]
    ]
    model = MobileNetV3(cfgs, mode='large', **kwargs)
    pretrained_dict = torch.load("./pretrained/mobilenetv3-large-1cd25616.pth")

    model_dict = model.state_dict()
    for name in pretrained_dict:
        if name not in model_dict:
            print(name, " not in model_dict!!!!!!!!!")
            continue
        else:
            print(name, "OK!!!")

        if name == "features.0.0.weight":
            model_dict[name][:, :3, :, :] = pretrained_dict[name]
            model_dict[name][:, 3:, :, :] = 0
            continue
        model_dict[name] = pretrained_dict[name]

    model.load_state_dict(model_dict)
    return model



class skipModule(nn.Module):
    def __init__(self):
        super(skipModule, self).__init__()

        self.skip_5 = nn.Sequential(
                OrderedDict([
                    ("conv1", nn.Conv2d(160, 64, 3, 1, 1, bias = False)),
                    ("norm1", nn.BatchNorm2d(64)),
                ])
                )

        self.skip_4 = nn.Sequential(
                OrderedDict([
                    ("conv1", nn.Conv2d(112, 64, 3, 1, 1, bias = False)),
                    ("norm1", nn.BatchNorm2d(64)),
                ])
                )

        self.skip_3 = nn.Sequential(
                OrderedDict([
                    ("conv1", nn.Conv2d(40, 64, 3, 1, 1, bias = False)),
                    ("norm1", nn.BatchNorm2d(64)),
                ])
                )

        self.skip_2 = nn.Sequential(
                OrderedDict([
                    ("conv1", nn.Conv2d(24, 64, 3, 1, 1, bias = False)),
                    ("norm1", nn.BatchNorm2d(64)),
                ])
                )

        self.skip_1 = nn.Sequential(
                OrderedDict([
                    ("conv1", nn.Conv2d(16, 64, 3, 1, 1, bias = False)),
                    ("norm1", nn.BatchNorm2d(64)),
                ])
                )
    def forward(self, inFeatures):
        out = {}
        out['stage5'] = self.skip_5 (inFeatures['stage5'])
        out['stage4'] = self.skip_4 (inFeatures['stage4'])
        out['stage3'] = self.skip_3 (inFeatures['stage3'])
        out['stage2'] = self.skip_2 (inFeatures['stage2'])
        out['stage1'] = self.skip_1 (inFeatures['stage1'])
        return out


class theModel(nn.Module):

    def __init__(self):
        super(theModel, self).__init__()

        self.backbone = mobilenetv3_large(image_channel = 4)
        self.skip = skipModule()

        decoder_channels_5 = 64
        decoder_channels_4 = 64
        decoder_channels_3 = 64
        decoder_channels_2 = 64
        decoder_channels_1 = 64
        decoder_channels_0 = 32

        self.decoder4 = nn.Sequential(
                OrderedDict([
                    ("prelu", nn.PReLU(64)),
                    ("conv", nn.Conv2d(64, 64, 3, 1, 1, bias = False)),
                    ("norm", nn.BatchNorm2d(64)),
                ])
                )

        self.decoder3 = nn.Sequential(
                OrderedDict([
                    ("prelu", nn.PReLU(64)),
                    ("conv", nn.Conv2d(64, 64, 3, 1, 1, bias = False)),
                    ("norm", nn.BatchNorm2d(64)),
                ])
                )

        self.decoder2 = nn.Sequential(
                OrderedDict([
                    ("prelu", nn.PReLU(64)),
                    ("conv", nn.Conv2d(64, 64, 3, 1, 1, bias = False)),
                    ("norm", nn.BatchNorm2d(64)),
                ])
                )

        self.decoder1 = nn.Sequential(
                OrderedDict([
                    ("prelu", nn.PReLU(64)),
                    ("conv", nn.Conv2d(64, 64, 3, 1, 1, bias = False)),
                    ("norm", nn.BatchNorm2d(64)),
                ])
                )

        self.stage1_64_to_32 = nn.Sequential(
                OrderedDict([
                    ("prelu", nn.PReLU(64)),
                    ("conv", nn.Conv2d(64, 32, 3, 1, 1, bias = False)),
                    ("norm", nn.BatchNorm2d(32)),
                ])
                )

        self.decoder_0 = nn.Sequential(
                OrderedDict([
                    ("prelu1", nn.PReLU(decoder_channels_0)),
                    ("conv1", nn.Conv2d(decoder_channels_0, decoder_channels_0, 3, 1, 1, bias = False)),
                    ("norm1", nn.BatchNorm2d(decoder_channels_0)),
                    ("prelu2", nn.PReLU(decoder_channels_0)),
                    ("conv2", nn.Conv2d(decoder_channels_0, 1, 1, 1, 0)),
                ])
                )
        self.skip_0 = nn.Sequential(
                OrderedDict([
                    ("conv1", nn.Conv2d(3, decoder_channels_0, 3, 1, 1, bias = False)),
                    ("norm1", nn.BatchNorm2d(decoder_channels_0)),
                    ("prelu1", nn.PReLU(decoder_channels_0)),
                    ("conv2", nn.Conv2d(decoder_channels_0, decoder_channels_0, 3, 1, 1, bias = False)),
                    ("norm2", nn.BatchNorm2d(decoder_channels_0)),
                ])
                )


    def forward(self, x):

        encoder_out = self.backbone(x)
        skip_out = self.skip(encoder_out)

        skip_out['stage4'] = skip_out['stage4'] + self.decoder4(F.interpolate(skip_out['stage5'], skip_out['stage4'].shape[2:], mode = "nearest"))
        skip_out['stage3'] = skip_out['stage3'] + self.decoder3(F.interpolate(skip_out['stage4'], skip_out['stage3'].shape[2:], mode = "nearest"))
        skip_out['stage2'] = skip_out['stage2'] + self.decoder2(F.interpolate(skip_out['stage3'], skip_out['stage2'].shape[2:], mode = "nearest"))
        skip_out['stage1'] = skip_out['stage1'] + self.decoder1(F.interpolate(skip_out['stage3'], skip_out['stage1'].shape[2:], mode = "nearest"))

        skip_out['stage1'] = self.stage1_64_to_32(skip_out['stage1'])

        x0_decoder = F.interpolate(skip_out['stage1'], x.shape[2:], mode = "bilinear")
        pred_alpha = self.decoder_0(self.skip_0(x[:,:3,:,:]) + x0_decoder)

        return pred_alpha



if __name__ == "__main__":

    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    model = mobilenetv3_large()
    model.eval()
    model.cuda()
    dump_x = torch.randn(1, 3, 2048, 2048).cuda()
    y = model(dump_x)
    pdb.set_trace()



