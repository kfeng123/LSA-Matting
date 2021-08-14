from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils import config
from .backbone.vovnet import VoVNet
from .backbone.resnet import resnet18, resnet50
from .backbone.resnet26d import resnet26d
from .backbone.mmclassification.resnet import ResNetV1d
from .skip.skip import skipModule_simple, skipModule
from .decoder.decoder import  decoderModule
from .spatial_path.spatial_path import spatial_path

class theModel(nn.Module):
    def __init__(self):
        super(theModel, self).__init__()
        self.lastStage = 5
        image_channel = 4
        self.backbone = ResNetV1d(in_channels = image_channel, depth = 50)
        #self.skip = skipModule_simple(self.backbone._out_feature_channels, lastStage = 5, image_channel = image_channel, ifPPM = True)
        self.skip = skipModule(self.backbone._out_feature_channels)
        self.decoder = decoderModule(self.skip.outChannels, lastStage = 5, image_channel = image_channel)
        self.spatial_path = spatial_path()

    def forward(self, x):
        encoder_out = self.backbone(x[:,:4,:,:])
        skip_out = self.skip(x, encoder_out)
        decoder_out = self.decoder( skip_out )
        fine_out = self.spatial_path(x[:,:3,:,:], decoder_out['feature'])
        out = {}
        out['alpha'] = fine_out['alpha']
        #out['alpha_coarse'] = decoder_out['alpha_coarse']
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


