from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from ...utils import config

class spatial_path(nn.Module):
    def __init__(self):
        super(spatial_path, self).__init__()
        self.add_module("encoder_0",
            nn.Sequential(OrderedDict([
                                ("conv1", nn.Conv2d( 3, 16, 3, 1, 1, bias = True)),
                                #("norm1", nn.BatchNorm2d(16)),
                                ("ReLU1", nn.ReLU(inplace = True)),
                            ]))
                )
        self.add_module("encoder_1",
            nn.Sequential(OrderedDict([
                                ("conv1", nn.Conv2d( 16, 32, 3, 2, 1, bias = True)),
                                #("norm1", nn.BatchNorm2d(32)),
                                ("ReLU1", nn.ReLU(inplace = True)),
                                ("conv2", nn.Conv2d( 32, 32, 3, 1, 1, bias = True)),
                                #("norm2", nn.BatchNorm2d(32)),
                                ("ReLU2", nn.ReLU(inplace = True)),
                                ("conv3", nn.Conv2d( 32, 64, 3, 1, 1, bias = True)),
                                ("ReLU3", nn.ReLU(inplace = True)),
                            ]))
                )

        self.add_module("encoder_2",
            nn.Sequential(OrderedDict([
                                ("conv1", nn.Conv2d( 64, 64, 3, 2, 1, bias = True)),
                                #("norm1", nn.BatchNorm2d(64)),
                                ("ReLU1", nn.ReLU(inplace = True)),
                                ("conv2", nn.Conv2d( 64, 64, 3, 1, 1, bias = True)),
                                #("norm2", nn.BatchNorm2d(64)),
                                ("ReLU2", nn.ReLU(inplace = True)),
                                ("conv3", nn.Conv2d( 64, 64, 3, 1, 1, bias = True)),
                                ("ReLU3", nn.ReLU(inplace = True)),
                            ]))
                )

        self.add_module("fuse_conv",
            nn.Sequential(OrderedDict([
                                ("conv1", nn.Conv2d( 128, 128, 3, 1, 1, bias = True)),
                                #("norm1", nn.BatchNorm2d(128)),
                                ("ReLU1", nn.ReLU(inplace = True)),
                            ]))
                )

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.pconv = nn.Sequential(OrderedDict([
                                ("conv1", nn.Conv2d( 128, 128, 3, 1, 1, bias = True)),
                                ("ReLU1", nn.ReLU(inplace = True)),
                                ("conv2", nn.Conv2d( 128, 128, 3, 1, 1, bias = True)),
                            ]))

        self.add_module("decoder_1",
            nn.Sequential(OrderedDict([
                                ("conv1", nn.Conv2d( 128 + 64, 32, 3, 1, 1, bias = True)),
                                #("norm1", nn.BatchNorm2d(32)),
                                ("ReLU1", nn.ReLU(inplace = True)),
                            ]))
                )

        self.add_module("decoder_0",
            nn.Sequential(OrderedDict([
                                ("conv1", nn.Conv2d( 32 + 16, 32, 3, 1, 1, bias = True)),
                                ("ReLU1", nn.ReLU(inplace = True)),
                                ("conv2", nn.Conv2d( 32, 32, 3, 1, 1, bias = True)),
                                ("ReLU2", nn.ReLU(inplace = True)),
                                ("conv3", nn.Conv2d( 32, 1, 1, 1, 0, bias = True)),
                            ]))
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

    def forward(self, image, feature):
        x0 = self.encoder_0(image)
        x1 = self.encoder_1(x0)
        x2 = self.encoder_2(x1)

        x2_fuse = torch.cat([x2, feature], 1)
        x2_fuse = self.fuse_conv(x2_fuse)
        w = self.avg_pool(x2_fuse)
        w = self.pconv(w)
        y2 = x2_fuse * (F.relu6(w + 3.0, inplace = True) / 6.0 + 1.0)
        y1 = F.interpolate(y2, x1.shape[2:], mode = "nearest")
        y1 = torch.cat([y1, x1], 1)
        y1 = self.decoder_1(y1)
        y0 = F.interpolate(y1, x0.shape[2:], mode = "nearest")
        y0 = torch.cat([y0, x0], 1)
        alpha = self.decoder_0(y0)
        out = {}
        out['alpha'] = alpha

        return out
