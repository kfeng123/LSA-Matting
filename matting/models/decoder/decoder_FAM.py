from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from ...utils import config

class my_conv(nn.Module):
    def __init__(self, inp, oup):
        super(my_conv, self).__init__()
        self.conv1 = nn.Conv2d(inp, oup, 3, 1, 1, bias = False)
        self.norm1 = nn.BatchNorm2d(oup)
        self.relu1 = nn.ReLU(inplace = True)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(oup, oup, 1, 1, 0)

    def forward(self, input):
        fea = self.conv1(input)
        fea = self.norm1(fea)
        fea = self.relu1(fea)

        w = self.avg_pool(fea)
        w = self.fc(w)
        return fea * ( F.relu6(w + 3.0, inplace = True) / 6.0 )

class FAM_module(nn.Module):
    def __init__(self, left_channels, down_channels, m_channels, out_channels):
        super(FAM_module, self).__init__()
        self.left_conv = nn.Conv2d(left_channels, m_channels, 1, bias=False)
        self.down_conv = nn.Conv2d(down_channels, m_channels, 1, bias=False)
        self.flow_make = nn.Conv2d(2 * m_channels, 2, kernel_size = 3, padding=1, bias=False)
        self.final_conv = nn.Conv2d(2 * m_channels, out_channels, kernel_size = 3, padding=1)

    def forward(self, left_feature, down_feature):
        down_feature = self.down_conv(down_feature)
        left_feature = self.left_conv(left_feature)
        left_shape = left_feature.shape[2], left_feature.shape[3]
        down_feature_upsampled = F.interpolate(down_feature, size = left_shape, mode="bilinear")
        flow = self.flow_make(torch.cat([down_feature_upsampled, left_feature], 1))
        feature = self.flow_warp(down_feature, flow, size = left_shape)
        feature = torch.cat([feature, left_feature], 1)
        return self.final_conv(feature)

    def flow_warp(self, input, flow, size):
        out_h, out_w = size
        n, c, h, w = input.size()
        norm = torch.tensor([[[[out_w, out_h]]]]).type_as(input).to(input.device)
        h = torch.linspace(-1.0, 1.0, out_h).view(out_h, 1).repeat(1, out_w)
        w = torch.linspace(-1.0, 1.0, out_w).view(1, out_w).repeat(out_h, 1)
        grid = torch.cat((w.unsqueeze(2), h.unsqueeze(2)), 2)
        grid = grid.repeat(n, 1, 1, 1).type_as(input).to(input.device)
        grid = grid + flow.permute(0, 2, 3, 1) / norm
        output = F.grid_sample(input, grid)
        return output

class decoderModule(nn.Module):
    def __init__(self, inChannels, lastStage = 4, image_channel = 4):
        super(decoderModule, self).__init__()
        self.lastStage = lastStage
        self.outChannels ={'stage0': 1,
                'stage1': 64,
                'stage2': 64,
                'stage3': 128,
                'stage4': 256,
                'stage5': 512,
                }
        self.inChannels = {'stage0': inChannels['stage0'] + self.outChannels['stage1'],
                'stage1': inChannels['stage1'] + self.outChannels['stage2'],
                'stage2': inChannels['stage2'] + self.outChannels['stage3'],
                'stage3': inChannels['stage3'] + self.outChannels['stage4'],
                'stage4': inChannels['stage4'] + self.outChannels['stage5'],
                'stage5': inChannels['stage5'],
                }
        for i in range(1, self.lastStage + 1):
            self.add_module("decoder_"+str(i),
                            my_conv(self.inChannels['stage'+str(i)], self.outChannels['stage'+str(i)])
                    )
        self.final_fusion = FAM_module(left_channels = image_channel, down_channels = self.outChannels['stage1'], m_channels = 32, out_channels = 32)
        self.final_final = nn.Sequential(
                OrderedDict([
                    ("relu1", nn.ReLU(inplace=True)),
                    ("conv1", nn.Conv2d(32, 32, 3, 1, 1)),
                    ("relu2", nn.ReLU(inplace=True)),
                    ("conv2", nn.Conv2d(32, 32, 3, 1, 1)),
                    ("relu3", nn.ReLU(inplace=True)),
                    ("conv3", nn.Conv2d(32, 1, 3, 1, 1)),
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
        #trimap = features['stage0'][:,3:4,:,:]
        out = {}
        theStages = list(range(2, self.lastStage + 1))[::-1]
        #for stage in theStages:
        #    tmp = getattr(self, "decoder_"+str(stage))(features['stage'+str(stage)])
        #    tmp = F.interpolate(tmp, features['stage'+str(stage-1)].shape[2:], mode = "nearest")
        #    features['stage'+str(stage-1)] = torch.cat([features['stage'+str(stage-1)], tmp], 1)

        # stage 5 to stage 4
        tmp = self.decoder_5(features['stage5'])
        tmp = F.interpolate(tmp, features['stage4'].shape[2:], mode = "nearest")
        tmp = torch.cat([features['stage4'], tmp], 1)
        # stage 4 to stage 3
        tmp = self.decoder_4(tmp)
        tmp = F.interpolate(tmp, features['stage3'].shape[2:], mode = "nearest")
        tmp = torch.cat([features['stage3'], tmp], 1)
        # stage 3 to stage 2
        tmp = self.decoder_3(tmp)
        tmp = F.interpolate(tmp, features['stage2'].shape[2:], mode = "nearest")
        tmp = torch.cat([features['stage2'], tmp], 1)
        # stage 2 to stage 1
        tmp = self.decoder_2(tmp)
        tmp = F.interpolate(tmp, features['stage1'].shape[2:], mode = "nearest")
        tmp = torch.cat([features['stage1'], tmp], 1)

        tmp = self.decoder_1(tmp)
        alpha = self.final_fusion(features['stage0'], tmp)
        alpha = self.final_final(alpha)
        out['alpha'] = alpha


        return out

