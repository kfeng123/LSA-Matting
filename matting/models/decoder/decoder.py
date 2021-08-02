from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from ...utils import config

class decoderModule(nn.Module):
    def __init__(self, inChannels, lastStage = 4, image_channel = 4):
        super(decoderModule, self).__init__()
        self.lastStage = lastStage
        self.outChannels ={'stage0': 1,
                'stage2': 64,
                'stage3': 128,
                'stage4': 256,
                'stage5': 512,
                }
        self.inChannels = {
                'stage2': inChannels['stage2'] + self.outChannels['stage3'],
                'stage3': inChannels['stage3'] + self.outChannels['stage4'],
                'stage4': inChannels['stage4'] + self.outChannels['stage5'],
                'stage5': inChannels['stage5'],
                }
        for i in range(2, 5 + 1):
            self.add_module("decoder_"+str(i),
                nn.Sequential(OrderedDict([
                                    ("conv1", nn.Conv2d( self.inChannels['stage'+str(i)], self.outChannels['stage'+str(i)], 3, 1, 1, bias = False)),
                                    ("norm1", nn.BatchNorm2d(self.outChannels['stage'+str(i)])),
                                    ("prelu1", nn.PReLU(self.outChannels['stage' + str(i)])),
                                ]))
                    )
        self.final_final = nn.Sequential(
                OrderedDict([
                    ("conv1", nn.Conv2d(32, 32, 3, 1, 1)),
                    ("relu2", nn.ReLU(inplace=True)),
                    ("conv2", nn.Conv2d(32, 1, 1, 1, 0)),
                ])
        )
        def init_weights(m):
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.fill_(0)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.fill_(0)
        self.apply(init_weights)

    def forward(self, features):
        trimap = features['stage0'][:,3:4,:,:]
        out = {}

        tmp = self.decoder_5(features['stage5'])
        tmp = F.interpolate(tmp, features['stage4'].shape[2:], mode = "nearest")
        features['stage4'] = torch.cat([features['stage4'], tmp], 1)

        tmp = self.decoder_4(features['stage4'])
        tmp = F.interpolate(tmp, features['stage3'].shape[2:], mode = "nearest")
        features['stage3'] = torch.cat([features['stage3'], tmp], 1)

        tmp = self.decoder_3(features['stage3'])
        tmp = F.interpolate(tmp, features['stage2'].shape[2:], mode = "nearest")
        features['stage2'] = torch.cat([features['stage2'], tmp], 1)

        alpha = self.final_final(features['stage2'])
        out['alpha'] = F.interpolate(alpha, features['stage0'].shape[2:], mode = "bilinear")

        return out

