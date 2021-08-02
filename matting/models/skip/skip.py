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




class skip_attention(nn.Module):
    def __init__(self, inp, oup):
        super(skip_attention, self).__init__()
        self.conv1 = nn.Conv2d(inp, oup, 3, 1, 1, bias = False)
        self.norm1 = nn.BatchNorm2d(oup)
        self.relu1 = nn.ReLU()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.pconv = nn.Conv2d(oup, oup, 1, 1, 0)

    def forward(self, x):
        fea = self.conv1(x)
        fea = self.norm1(fea)
        fea = self.relu1(fea)

        w = self.avg_pool(fea)
        w = self.pconv(w)
        return fea * (F.relu6(w + 3.0, inplace = True) / 6.0)

class trimap_process(nn.Module):
    def __init(self):
        super(trimap_process, self).__init__()
        self.relu = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.maxpool4 = nn.MaxPool2d(kernel_size = 4, stride = 4)
        self.maxpool8 = nn.MaxPool2d(kernel_size = 8, stride = 8)
        self.maxpool16 = nn.MaxPool2d(kernel_size = 16, stride = 16)
        self.maxpool32 = nn.MaxPool2d(kernel_size = 32, stride = 32)

    def forward(self, image):
        trimap = image[:,3,:,:]
        trimap = trimap[:, None, :, :]
        F_mask = self.relu(trimap - 3./4.) * 4
        B_mask = self.relu(1./4. - trimap) * 4
        trimaps = {}
        trimaps['stage2_F_large'] = self.maxpool4(F_mask)
        trimaps['stage3_F_large'] = self.maxpool2(trimaps['stage2_F_large'])
        trimaps['stage4_F_large'] = self.maxpool2(trimaps['stage3_F_large'])
        trimaps['stage5_F_large'] = self.maxpool2(trimaps['stage4_F_large'])

        trimaps['stage2_F_small'] = self.maxpool4(1-F_mask)
        trimaps['stage3_F_small'] = self.maxpool2(trimaps['stage2_F_small'])
        trimaps['stage4_F_small'] = self.maxpool2(trimaps['stage3_F_small'])
        trimaps['stage5_F_small'] = self.maxpool2(trimaps['stage4_F_small'])
        trimaps['stage2_F_small'] = 1 - trimaps['stage2_F_small']
        trimaps['stage3_F_small'] = 1 - trimaps['stage3_F_small']
        trimaps['stage4_F_small'] = 1 - trimaps['stage4_F_small']
        trimaps['stage5_F_small'] = 1 - trimaps['stage5_F_small']

        trimaps['stage2_B_large'] = self.maxpool4(B_mask)
        trimaps['stage3_B_large'] = self.maxpool2(trimaps['stage2_B_large'])
        trimaps['stage4_B_large'] = self.maxpool2(trimaps['stage3_B_large'])
        trimaps['stage5_B_large'] = self.maxpool2(trimaps['stage4_B_large'])

        trimaps['stage2_B_small'] = self.maxpool4(1-B_mask)
        trimaps['stage3_B_small'] = self.maxpool2(trimaps['stage2_B_small'])
        trimaps['stage4_B_small'] = self.maxpool2(trimaps['stage3_B_small'])
        trimaps['stage5_B_small'] = self.maxpool2(trimaps['stage4_B_small'])
        trimaps['stage2_B_small'] = 1 - trimaps['stage2_B_small']
        trimaps['stage3_B_small'] = 1 - trimaps['stage3_B_small']
        trimaps['stage4_B_small'] = 1 - trimaps['stage4_B_small']
        trimaps['stage5_B_small'] = 1 - trimaps['stage5_B_small']

        out = {}
        out['stage0'] = torch.cat([F_mask, B_mask], 1)
        out['stage2'] = torch.cat([trimaps['stage2_F_small'], trimaps['stage2_F_large'], trimaps['stage2_B_small'], trimaps['stage2_B_large']], 1)
        out['stage3'] = torch.cat([trimaps['stage3_F_small'], trimaps['stage3_F_large'], trimaps['stage3_B_small'], trimaps['stage3_B_large']], 1)
        out['stage4'] = torch.cat([trimaps['stage4_F_small'], trimaps['stage4_F_large'], trimaps['stage4_B_small'], trimaps['stage4_B_large']], 1)
        out['stage5'] = torch.cat([trimaps['stage5_F_small'], trimaps['stage5_F_large'], trimaps['stage5_B_small'], trimaps['stage5_B_large']], 1)
        return out


class skipModule(nn.Module):
    def __init__(self, inChannels):
        super(skipModule, self).__init__()
        self.outChannels ={
            'stage0': 32,
            'stage1': 64,
            'stage2': 64,
            'stage3': 128,
            'stage4': 256,
            'stage5': 512,
        }
        #self.skip0 = skip_attention(3, self.outChannels['stage0'])
        #self.skip1 = skip_attention(inChannels['stage1'] + 4, self.outChannels['stage1'])
        self.skip2 = skip_attention(inChannels['stage2'] + 4, self.outChannels['stage2'])
        self.skip3 = skip_attention(inChannels['stage3'] + 4, self.outChannels['stage3'])
        self.skip4 = skip_attention(inChannels['stage4'] + 4, self.outChannels['stage4'])
        self.skip5 = skip_attention(inChannels['stage5'] + 4, self.outChannels['stage5'])
        self.trimap_process = trimap_process()

        # initialize weights
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
        trimaps = self.trimap_process(image)
        out = {}
        out['stage0'] = image[:,:3,:,:]
        out['stage1'] = inFeatures['stage1']
        out['stage2'] = self.skip2(torch.cat([inFeatures['stage2'], trimaps['stage2']], 1))
        out['stage3'] = self.skip3(torch.cat([inFeatures['stage3'], trimaps['stage3']], 1))
        out['stage4'] = self.skip4(torch.cat([inFeatures['stage4'], trimaps['stage4']], 1))
        out['stage5'] = self.skip5(torch.cat([inFeatures['stage5'], trimaps['stage5']], 1))
        return out
