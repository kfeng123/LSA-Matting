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

class skipModule(nn.Module):
    def __init__(self, inChannels, lastStage = 4, image_channel = 4, ifPPM = True):
        super(skipModule, self).__init__()

        self.lastStage = lastStage

        self.outChannels ={'stage0': image_channel,
                'stage1': 128,
                'stage2': 128,
                'stage3': 128,
                'stage4': 256,
                'stage5': 256,
                }

        for i in range(1, self.lastStage + 1):
            self.add_module("skip_"+str(i),
                    nn.Sequential(OrderedDict([
                                    ("conv1", nn.Conv2d(inChannels['stage'+str(i)], self.outChannels['stage'+str(i)], 3, 1, 1, bias = False)),
                                    ("norm1", nn.BatchNorm2d(self.outChannels['stage'+str(i)])),
                                    ]))
                    )

        #self.skip_0 = nn.Sequential(
        #        OrderedDict([
        #            ("conv1", nn.Conv2d(image_channel, self.outChannels['stage0'], 3, 1, 1, bias = False)),
        #            ("norm1", nn.BatchNorm2d(self.outChannels['stage0'])),
        #        ])
        #        )

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
            out['stage' + str(i)] = getattr(self, "skip_"+str(i))(inFeatures['stage'+str(i)])

        if self.ifPPM:
            out['stage' + str(self.lastStage)] = self.ppm(out['stage'+str(self.lastStage)])

        #out['stage0'] = self.skip_0(image)
        out['stage0'] = image

        return out



class skipModule_simple(nn.Module):
    def __init__(self, inChannels, lastStage = 4, image_channel = 4, ifPPM = True):
        super(skipModule_simple, self).__init__()

        self.lastStage = lastStage

        self.outChannels ={'stage0': image_channel,
                #'stage1': inChannels['stage1'],
                #'stage2': inChannels['stage2'],
                #'stage3': inChannels['stage3'],
                #'stage4': inChannels['stage4'],
                #'stage5': inChannels['stage5'],
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

        #out['stage0'] = self.skip_0(image)
        out['stage0'] = image

        return out



