from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils import config
from .backbone.mmclassification.resnet import ResNetV1d
from .skip.skip import skipModule_simple
from .decoder.decoder_FAM import  decoderModule

class theModel(nn.Module):
    def __init__(self):
        super(theModel, self).__init__()
        self.lastStage = 5
        image_channel = 4
        self.backbone = ResNetV1d(in_channels = image_channel, depth = 50)
        self.skip = skipModule_simple(self.backbone._out_feature_channels, lastStage = 5, image_channel = image_channel, ifPPM = False)
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
        self.a = torch.zeros([1, simple_model.skip.outChannels['stage5'], 1, 1]).cuda()
        self.a.requires_grad = True
        self.b = torch.zeros([1, simple_model.skip.outChannels['stage5'], 1, 1]).cuda()
        self.b.requires_grad = True
        # laplacian kernel
        self.laplacian_kernel = torch.ones((1,1,3,3), requires_grad = False).cuda()
        self.laplacian_kernel[0,0,1,1] = -8
    def forward(self, x):
        nn.init.zeros_(self.a)
        nn.init.zeros_(self.b)

        trimap_detach = x[:,3:4,:,:].detach()
        pos_detach = (trimap_detach > 0.75) * 1.
        neg_detach = (trimap_detach < 0.25) * 1.
        unknown_detach = 1 - pos_detach - neg_detach
        pos_edge_detach = F.conv2d(pos_detach, self.laplacian_kernel, padding = 1)
        pos_edge_detach = torch.abs(pos_edge_detach)
        pos_edge_detach = pos_edge_detach * unknown_detach
        pos_pixel_number = pos_edge_detach.sum() + 0.1

        neg_edge_detach = F.conv2d(neg_detach, self.laplacian_kernel, padding = 1)
        neg_edge_detach = torch.abs(neg_edge_detach)
        neg_edge_detach = neg_edge_detach * unknown_detach
        neg_pixel_number = neg_edge_detach.sum() + 0.1

        with torch.no_grad():
            encoder_out = self.backbone(x)
            skip_out = self.skip(x, encoder_out)

        with torch.enable_grad():
            tmp_stage5 = skip_out['stage5']
            for the_step in range(5):
                skip_out['stage5'] = tmp_stage5 * torch.sigmoid(self.a) + self.b
                decoder_out = self.decoder( skip_out )
                decoder_out['alpha'] = torch.clamp(decoder_out['alpha'], 0, 1)
                loss = (1 - decoder_out['alpha'] * pos_edge_detach).sum() / pos_pixel_number + \
                (decoder_out['alpha'] * neg_edge_detach).sum() / neg_pixel_number
                print("haha loss ", the_step, ":", loss)

                with torch.no_grad():
                    if self.a.grad is not None:
                        self.a.grad.detach().zero_()
                    if self.b.grad is not None:
                        self.b.grad.detach().zero_()

                loss.backward()

                with torch.no_grad():
                    self.a.add_( self.a.grad, alpha = - 1e-1)
                    self.b.add_( self.b.grad, alpha = - 1e-1)

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


