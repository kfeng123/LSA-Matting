from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils import config
from .backbone.mmclassification.resnet import ResNetV1d
from .skip.skip_aspp import skipModule_simple
from .decoder.decoder_simple import  decoderModule

class theModel(nn.Module):
    def __init__(self):
        super(theModel, self).__init__()
        self.lastStage = 5
        image_channel = 4
        self.backbone = ResNetV1d(in_channels = image_channel, depth = 50)
        self.skip = skipModule_simple(self.backbone._out_feature_channels, image_channel = image_channel, ifASPP = False)
        self.decoder = decoderModule(self.skip.outChannels, lastStage = 5, image_channel = image_channel)

    def forward(self, x):
        #encoder_out = self.backbone(x[:,:4,:,:])
        encoder_out = self.backbone(x)
        skip_out = self.skip(x, encoder_out)
        decoder_out = self.decoder( skip_out )
        out = {}
        out['alpha'] = decoder_out['alpha']
        return out


class test_time_model(nn.Module):
    def __init__(self, simple_model):
        super(test_time_model, self).__init__()
        # model
        self.backbone = simple_model.backbone
        self.skip = simple_model.skip
        self.decoder = simple_model.decoder
        # hyperparameter to be optimized
        self.hyper_stages = [4]
        self.A = {}
        self.B = {}
        #for i in self.hyper_stages:
        #    self.A[ 'stage' + str(i) ] = torch.zeros([1, simple_model.skip.outChannels[ 'stage' + str(i) ], 16, 16]).cuda()
        #    self.A[ 'stage' + str(i) ].requires_grad = True
        #    self.B[ 'stage' + str(i) ] = torch.zeros([1, simple_model.skip.outChannels[ 'stage' + str(i) ], 16, 16]).cuda()
        #    self.B[ 'stage' + str(i) ].requires_grad = True

        self.the_channels = {
                'stage0': 1,
                'stage1': 64,
                'stage2': 64,
                'stage3': 128,
                'stage4': 256,
                'stage5': 512,
                }

        for i in self.hyper_stages:
            self.A[ 'stage' + str(i) ] = torch.zeros([1, self.the_channels[ 'stage' + str(i) ], 16, 16]).cuda()
            self.A[ 'stage' + str(i) ].requires_grad = True
            self.B[ 'stage' + str(i) ] = torch.zeros([1, self.the_channels[ 'stage' + str(i) ], 16, 16]).cuda()
            self.B[ 'stage' + str(i) ].requires_grad = True
        #self.A[ 'stage4'] = torch.zeros([1, 256, 16, 16]).cuda()
        #self.A[ 'stage4'].requires_grad = True
        #self.B[ 'stage4'] = torch.zeros([1, 256, 16, 16]).cuda()
        #self.B[ 'stage4'].requires_grad = True

        #self.A[ 'stage3'] = torch.zeros([1, 128, 16, 16]).cuda()
        #self.A[ 'stage3'].requires_grad = True
        #self.B[ 'stage3'] = torch.zeros([1, 128, 16, 16]).cuda()
        #self.B[ 'stage3'].requires_grad = True

        # laplacian kernel
        self.laplacian_kernel = torch.ones((1,1,3,3), requires_grad = False).cuda()
        self.laplacian_kernel[0,0,1,1] = -8
    def forward(self, x):
        _, _, h, w = x.shape

        for i in self.hyper_stages:
            nn.init.zeros_(self.A['stage'+str(i)])
            nn.init.zeros_(self.B['stage'+str(i)])

        trimap_detach = x[:,3:4,:,:].detach()
        pos_detach = (trimap_detach > 0.75) * 1.
        neg_detach = (trimap_detach < 0.25) * 1.
        unknown_detach = 1 - pos_detach - neg_detach
        pos_edge_detach = F.conv2d(pos_detach, self.laplacian_kernel, padding = 1)
        pos_edge_detach = torch.abs(pos_edge_detach)
        pos_edge_detach = (pos_edge_detach > 0) * unknown_detach
        pos_pixel_number = pos_edge_detach.sum() + 0.1

        neg_edge_detach = F.conv2d(neg_detach, self.laplacian_kernel, padding = 1)
        neg_edge_detach = torch.abs(neg_edge_detach)
        neg_edge_detach = (neg_edge_detach > 0) * unknown_detach
        neg_pixel_number = neg_edge_detach.sum() + 0.1

        with torch.no_grad():
            encoder_out = self.backbone(x)
            skip_out = self.skip(x, encoder_out)

        with torch.enable_grad():
            skip_out_orig = {}
            for i in self.hyper_stages:
                skip_out_orig['stage' + str(i)] =skip_out['stage' + str(i)]

            original_alpha = self.decoder( skip_out )['alpha']
            for the_step in range(100):

                #for i in self.hyper_stages:
                #    A = F.interpolate(self.A['stage'+str(i)], skip_out_orig['stage'+str(i)].shape[2:], mode = "bicubic")
                #    B = F.interpolate(self.B['stage'+str(i)], skip_out_orig['stage'+str(i)].shape[2:], mode = "bicubic")
                #    #skip_out['stage'+str(i)] = skip_out_orig['stage'+str(i)] * torch.sigmoid(A) * 2 + B
                #    skip_out['stage'+str(i)] = skip_out_orig['stage'+str(i)] * torch.exp(A) + B

                #decoder_out = self.decoder( skip_out )

                # stage 5 to stage 4
                tmp = self.decoder.decoder_5(skip_out['stage5'])
                if 5 in self.hyper_stages:
                    A = F.interpolate(self.A['stage5'], skip_out['stage5'].shape[2:], mode = "bicubic")
                    B = F.interpolate(self.B['stage5'], skip_out['stage5'].shape[2:], mode = "bicubic")
                    tmp = tmp * torch.exp(A) + B

                tmp = F.interpolate(tmp, skip_out['stage4'].shape[2:], mode = "nearest")
                tmp = torch.cat([skip_out['stage4'], tmp], 1)
                # stage 4 to stage 3
                tmp = self.decoder.decoder_4(tmp)
                if 4 in self.hyper_stages:
                    A = F.interpolate(self.A['stage4'], skip_out['stage4'].shape[2:], mode = "bicubic")
                    B = F.interpolate(self.B['stage4'], skip_out['stage4'].shape[2:], mode = "bicubic")
                    tmp = tmp * torch.exp(A) + B
                tmp = F.interpolate(tmp, skip_out['stage3'].shape[2:], mode = "nearest")
                tmp = torch.cat([skip_out['stage3'], tmp], 1)
                # stage 3 to stage 2
                tmp = self.decoder.decoder_3(tmp)
                if 3 in self.hyper_stages:
                    A = F.interpolate(self.A['stage3'], skip_out['stage3'].shape[2:], mode = "bicubic")
                    B = F.interpolate(self.B['stage3'], skip_out['stage3'].shape[2:], mode = "bicubic")
                    tmp = tmp * torch.exp(A) + B
                tmp = F.interpolate(tmp, skip_out['stage2'].shape[2:], mode = "nearest")
                tmp = torch.cat([skip_out['stage2'], tmp], 1)
                # stage 2 to stage 1
                tmp = self.decoder.decoder_2(tmp)
                if 2 in self.hyper_stages:
                    A = F.interpolate(self.A['stage2'], skip_out['stage2'].shape[2:], mode = "bicubic")
                    B = F.interpolate(self.B['stage2'], skip_out['stage2'].shape[2:], mode = "bicubic")
                    tmp = tmp * torch.exp(A) + B
                tmp = F.interpolate(tmp, skip_out['stage1'].shape[2:], mode = "nearest")
                tmp = torch.cat([skip_out['stage1'], tmp], 1)

                tmp = self.decoder.decoder_1(tmp)
                if 1 in self.hyper_stages:
                    A = F.interpolate(self.A['stage1'], skip_out['stage1'].shape[2:], mode = "bicubic")
                    B = F.interpolate(self.B['stage1'], skip_out['stage1'].shape[2:], mode = "bicubic")
                    tmp = tmp * torch.exp(A) + B
                tmp = F.interpolate(tmp, skip_out['stage0'].shape[2:], mode = "bilinear")
                tmp = torch.cat([skip_out['stage0'], tmp], 1)
                decoder_out = {}
                decoder_out['alpha'] = self.decoder.final_final(tmp)

                decoder_out['alpha'] = torch.clamp(decoder_out['alpha'], 0, 1)
                #loss_edge = ((1 - decoder_out['alpha']) ** 2 * pos_edge_detach).sum() / pos_pixel_number + \
                #(decoder_out['alpha'] ** 2 * neg_edge_detach).sum() / neg_pixel_number
                loss_edge = ( ((1 - decoder_out['alpha']) ** 2 * pos_edge_detach).sum() + (decoder_out['alpha'] ** 2 * neg_edge_detach).sum() ) / (pos_pixel_number + neg_pixel_number)

                loss_preserve = (torch.abs( decoder_out['alpha'] - original_alpha.detach() ) * unknown_detach).sum() / unknown_detach.sum()

                loss = loss_edge + 0.1 * loss_preserve ** 2
                if the_step % 10 == 0:
                    print("Step ", the_step, ":", "Total Loss", loss.item(), "Edge loss: ", loss_edge.item(), "Preservation loss: ", loss_preserve.item())

                with torch.no_grad():
                    for i in self.hyper_stages:
                        if self.A['stage'+str(i)].grad is not None:
                            self.A['stage'+str(i)].grad.detach().zero_()
                        if self.B['stage'+str(i)].grad is not None:
                            self.B['stage'+str(i)].grad.detach().zero_()
                loss.backward()

                lr = 50
                with torch.no_grad():
                    for i in self.hyper_stages:
                        self.A['stage'+str(i)].add_( self.A['stage'+str(i)].grad, alpha = - lr)
                        self.B['stage'+str(i)].add_( self.B['stage'+str(i)].grad, alpha = - lr)

        out = {}
        out['alpha'] = decoder_out['alpha']
        return out

if __name__ == "__main__":

    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '6'
    model = theModel()
    model.eval()
    model.cuda()
    dump_x = torch.randn(1, 4, 2048, 2048).cuda()
    y = model(dump_x)
    import pdb
    pdb.set_trace()


