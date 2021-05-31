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
                    ("prelu", nn.PReLU(in_dim//4))
                ])))
    def forward(self, x):
        x_size = x.size()
        out = [x]
        for scale in self.scales:
            out.append(F.interpolate(getattr(self, "ppm"+str(scale))(x), x_size[2:], mode = "bilinear", align_corners = False))
        return torch.cat(out, 1)

class skipModule_simple(nn.Module):
    def __init__(self, inChannels, lastStage = 4, image_channel = 4, ifPPM = True):
        super(skipModule_simple, self).__init__()

        self.lastStage = lastStage

        self.outChannels ={'stage0': image_channel,
                }

        for i in range(1, self.lastStage + 1):
            self.outChannels['stage' + str(i)] = inChannels['stage' + str(i)]

        # PPM
        self.ifPPM = ifPPM
        if self.ifPPM:
            self.ppm = PPM(self.outChannels['stage'+str(self.lastStage)])
            self.outChannels['stage' + str(self.lastStage)] = self.outChannels['stage' + str(self.lastStage)] *2

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
        for i in range(1, self.lastStage + 1):
            out['stage' + str(i)] = inFeatures['stage'+str(i)]
        if self.ifPPM:
            out['stage' + str(self.lastStage)] = self.ppm(out['stage'+str(self.lastStage)])
        out['stage0'] = image
        return out
