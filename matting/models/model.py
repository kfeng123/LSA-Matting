from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils import config
from .backbone.vovnet import VoVNet
from .backbone.resnet import resnet18, resnet50
from .backbone.resnet26d import resnet26d
from .backbone.mmclassification.resnet import ResNetV1d
from .skip.skip import skipModule_simple
from .decoder.decoder_FAM import  decoderModule

class theModel(nn.Module):
    def __init__(self):
        super(theModel, self).__init__()
        self.lastStage = 5
        image_channel = 4
        #self.backbone = VoVNet(image_channel, myArch = "V-39-eSE", lastStage = 5)
        #self.backbone = resnet18(inChannel = image_channel)
        self.backbone = ResNetV1d(in_channels = image_channel, depth = 50)
        #self.backbone = resnet26d(inChannel = image_channel)
        self.skip = skipModule_simple(self.backbone._out_feature_channels, lastStage = 5, image_channel = image_channel, ifPPM = True)
        self.decoder = decoderModule(self.skip.outChannels, lastStage = 5, image_channel = image_channel)
    def forward(self, x):
        encoder_out = self.backbone(x)
        skip_out = self.skip(x, encoder_out)
        pred_alpha = self.decoder( skip_out )
        return pred_alpha

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


