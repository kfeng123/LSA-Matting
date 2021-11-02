from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F

class PPM(nn.Module):
    def __init__(self, in_dim):
        super(PPM, self).__init__()
        self.scales = [1, 2, 3, 6]
        for scale in self.scales:
            self.add_module("ppm"+str(scale), nn.Sequential(OrderedDict([
                    ("aap", nn.AdaptiveAvgPool2d(scale)),
                    ("conv", nn.Conv2d(in_dim, in_dim//4, kernel_size = 1, bias = False)),
                    ("norm", nn.BatchNorm2d(in_dim//4)),
                    ("prelu", nn.ReLU(inplace = True))
                ])))
    def forward(self, x):
        x_size = x.size()
        out = [x]
        for scale in self.scales:
            out.append(F.interpolate(getattr(self, "ppm"+str(scale))(x), x_size[2:], mode = "bilinear", align_corners = False))
        return torch.cat(out, 1)

# ASPP code is from torchvision
class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ]
        super(ASPPConv, self).__init__(*modules)


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())

    def forward(self, x):
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates, out_channels=256):
        super(ASPP, self).__init__()
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()))

        rates = tuple(atrous_rates)
        for rate in rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))

        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5))

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)

class skipModule_simple(nn.Module):
    def __init__(self, inChannels, image_channel = 4, ifASPP = True):
        super(skipModule_simple, self).__init__()

        self.outChannels ={'stage0': image_channel,
                }

        for i in range(1, 5 + 1):
            self.outChannels['stage' + str(i)] = inChannels['stage' + str(i)]

        # ASPP
        self.ifASPP = ifASPP
        if self.ifASPP:
            #ASPP(in_channels, [12, 24, 36])
            self.aspp5 = ASPP(self.outChannels['stage5'], [3, 6, 9], self.outChannels['stage5'])
            self.aspp4 = ASPP(self.outChannels['stage4'], [6, 12, 18], self.outChannels['stage4'])
            self.aspp3 = ASPP(self.outChannels['stage3'], [12, 24, 36], self.outChannels['stage3'])

        def init_weights(m):
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.fill_(0)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.fill_(0)
        self.apply(init_weights)

    def forward(self, image, inFeatures):
        out = {}
        for i in range(1, 5 + 1):
            out['stage' + str(i)] = inFeatures['stage'+str(i)]
            if self.ifASPP and i >= 3:
                out['stage' + str(i)] = getattr(self, 'aspp'+ str(i)) (inFeatures['stage'+str(i)])

        out['stage0'] = image
        return out

