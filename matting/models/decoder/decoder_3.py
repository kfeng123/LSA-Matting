from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from ...utils import config

class decoderModule(nn.Module):
    def __init__(self, inChannels, lastStage = 3, image_channel = 4, aux_loss = False, aux_loss_Urysohn = False):
        super(decoderModule, self).__init__()

        self.lastStage = lastStage

        self.aux_loss = aux_loss

        self.aux_loss_Urysohn = aux_loss_Urysohn

        if self.aux_loss_Urysohn:
            inChannels['stage0'] = inChannels['stage0'] + 1

        self.outChannels ={'stage0': 1,
                'stage1': 64,
                'stage2': 128,
                'stage3': 256,
                }
        self.inChannels = {'stage0': inChannels['stage0'] + self.outChannels['stage1'],
                'stage1': inChannels['stage1'] + self.outChannels['stage2'],
                'stage2': inChannels['stage2'] + self.outChannels['stage3'],
                'stage3': inChannels['stage3'],
                }


        for i in range(1, self.lastStage + 1):
            self.add_module("decoder_"+str(i),
                nn.Sequential(OrderedDict([
                                    ("conv1", nn.Conv2d( self.inChannels['stage'+str(i)], self.outChannels['stage'+str(i)], 3, 1, 1, bias = False)),
                                    ("norm1", nn.BatchNorm2d(self.outChannels['stage'+str(i)])),
                                    ("prelu1", nn.PReLU(self.outChannels['stage' + str(i)])),
                                ]))
                    )

        if config.spatial_propagation:
            self.decoder_0 = nn.Sequential(
                    OrderedDict([
                        ("conv1", nn.Conv2d(self.inChannels['stage0'], 32, 3, 1, 1, bias = True)),
                        #("norm1", nn.BatchNorm2d(32)),
                        ("prelu2", nn.PReLU(32)),
                        ("conv2", nn.Conv2d(32, 32, 3, 1, 1, bias = True)),
                        #("norm2", nn.BatchNorm2d(32)),
                        ("prelu3", nn.PReLU(32)),
                        ("conv3_sp", nn.Conv2d(32, 10, 3, 1, 1, bias = True)),
                    ])
                    )
            self.padding = nn.ReflectionPad2d(1)
        elif config.spatial_markov:
            self.decoder_0 = nn.Sequential(
                    OrderedDict([
                        ("conv1", nn.Conv2d(self.inChannels['stage0'], 32, 3, 1, 1, bias = True)),
                        ("prelu2", nn.PReLU(32)),
                        ("conv2", nn.Conv2d(32, 32, 3, 1, 1, bias = True)),
                        ("prelu3", nn.PReLU(32)),
                        ("conv3_sp", nn.Conv2d(32, 10, 3, 1, 1, bias = True)),
                    ])
                    )
            self.padding = nn.ReflectionPad2d(1)
        else:
            self.decoder_0 = nn.Sequential(
                    OrderedDict([
                        ("conv1", nn.Conv2d(self.inChannels['stage0'], 32, 3, 1, 1, bias = True)),
                        #("norm1", nn.BatchNorm2d(32)),
                        ("prelu2", nn.PReLU(32)),
                        ("conv2", nn.Conv2d(32, 32, 3, 1, 1, bias = True)),
                        #("norm2", nn.BatchNorm2d(32)),
                        ("prelu3", nn.PReLU(32)),
                        ("conv3", nn.Conv2d(32, 1, 3, 1, 1, bias = True)),
                    ])
                    )


        if self.aux_loss:
            self.decoder_aux = nn.Sequential(
                    OrderedDict([
                        ("conv1", nn.Conv2d(320, 320, 3, 1, 1, bias = False)),
                        ("norm1", nn.BatchNorm2d(320)),
                        ("prelu2", nn.PReLU(320)),
                        ("conv2", nn.Conv2d(320, 3, 3, 1, 1)),
                    ])
                    )
        if self.aux_loss_Urysohn:
            self.decoder_aux_Urysohn = nn.Sequential(
                    OrderedDict([
                        ("conv1", nn.Conv2d(self.inChannels['stage2'], 64, 3, 1, 1, bias = False)),
                        ("norm1", nn.BatchNorm2d(64)),
                        ("prelu1", nn.PReLU(64)),
                        ("conv2", nn.Conv2d(64, 64, 3, 1, 1, bias = False)),
                        ("norm2", nn.BatchNorm2d(64)),
                        ("prelu2", nn.PReLU(64)),
                        ("conv3", nn.Conv2d(64, 1, 3, 1, 1)),
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
        theStages = list(range(2, self.lastStage + 1))[::-1]

        for stage in theStages:
            tmp = getattr(self, "decoder_"+str(stage))(features['stage'+str(stage)])
            tmp = F.interpolate(tmp, features['stage'+str(stage-1)].shape[2:], mode = "nearest")
            features['stage'+str(stage-1)] = torch.cat([features['stage'+str(stage-1)], tmp], 1)

        tmp = self.decoder_1(features['stage1'])
        #tmp = F.pixel_shuffle(tmp, 2)
        tmp = F.interpolate(tmp, features['stage0'].shape[2:], mode = "bilinear")

        features['stage0'] = torch.cat([features['stage0'], tmp], 1)

        if self.aux_loss_Urysohn:
            aux_Urysohn = self.decoder_aux_Urysohn(features['stage2'])
            aux_Urysohn = F.interpolate(aux_Urysohn, features['stage0'].shape[2:], mode = "bilinear")
            out['aux_Urysohn'] = aux_Urysohn
            features['stage0'] = torch.cat([features['stage0'], aux_Urysohn], 1)
        else:
            out['aux_Urysohn'] = None

        alpha = self.decoder_0(features['stage0'])

        if config.spatial_propagation:
            spatial_weight = alpha[:,1:,:,:]
            spatial_weight = F.softmax(spatial_weight, dim = 1)
            #######################################
            #kappa_sum = spatial_weight.abs().sum(dim = 1, keepdim = True) - spatial_weight[:,4:5, :, :].abs()
            #spatial_weight = spatial_weight / kappa_sum
            #spatial_weight[:,4:5,:,:] = 1- (spatial_weight.sum(dim=1, keepdim = True) - spatial_weight[:,4:5, :, :])
            #######################################
            alpha = alpha[:,:1,:,:]
            assert(alpha.shape == trimap.shape)
            trimap_1 = F.relu(trimap -2/3) * 30
            trimap_0 = F.relu( 1/3 - trimap) * 30
            alpha = torch.clamp(alpha + trimap_1 - trimap_0, 0, 1)
            n, c, h, w = alpha.shape
            for _ in range(config.propagation_iteration):
                tmp_list = []
                alpha = self.padding(alpha)
                for i in range(3):
                    for j in range(3):
                        tmp_list.append(alpha[:,:,i:(i+h),j:(j+w)] * spatial_weight[:,i * 3 + j,:,:].unsqueeze(1) )
                alpha = torch.cat(tmp_list, dim = 1).sum(dim = 1, keepdim = True)
        elif config.spatial_markov:
            spatial_weight = alpha[:,1:,:,:]
            spatial_weight = F.softmax(spatial_weight, dim = 1)
            alpha = alpha[:,:1,:,:]
            assert(alpha.shape == trimap.shape)
            trimap_1 = F.relu(trimap -2/3) * 30
            trimap_0 = F.relu( 1/3 - trimap) * 30
            alpha = torch.clamp(alpha + trimap_1 - trimap_0, 0, 1)
            n, c, h, w = alpha.shape
            for _ in range(config.markov_iteration):
                tmp_alpha = alpha * spatial_weight
                tmp_alpha = self.padding(tmp_alpha)
                tmp_list = []
                torch.roll(alpha, shifts = (-1, -1), dims = (2, 3))
                torch.roll(alpha, shifts = (-1, 0), dims = (2, 3))
                torch.roll(alpha, shifts = (-1, 1), dims = (2, 3))
                for i in range(3):
                    for j in range(3):
                        tmp = torch.roll(tmp_alpha[:, i *3 + j, :, :].unsqueeze(1), shifts = (i-1, j-1), dims = (2, 3))
                        tmp_list.append(tmp)
                alpha = torch.cat(tmp_list, dim = 1).sum(dim = 1, keepdim = True)[:,:,1:(h+1),1:(w+1)]

        out['alpha'] = alpha

        if self.aux_loss:
            aux_alpha = self.decoder_aux(features['stage2'])
            aux_alpha = F.interpolate(aux_alpha, features['stage0'].shape[2:], mode = "nearest")
            out['aux_alpha'] = aux_alpha
        else:
            out['aux_alpha'] = None


        return out

