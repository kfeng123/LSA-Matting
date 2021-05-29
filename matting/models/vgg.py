import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import pdb
pretrained_keys = [
        'features.0.weight',
        'features.0.bias',
        'features.1.weight',
        'features.1.bias',
        'features.1.running_mean',
        'features.1.running_var',
        'features.3.weight',
        'features.3.bias',
        'features.4.weight',
        'features.4.bias',
        'features.4.running_mean',
        'features.4.running_var',
        'features.7.weight',
        'features.7.bias',
        'features.8.weight',
        'features.8.bias',
        'features.8.running_mean',
        'features.8.running_var',
        'features.10.weight',
        'features.10.bias',
        'features.11.weight',
        'features.11.bias',
        'features.11.running_mean',
        'features.11.running_var',
        'features.14.weight',
        'features.14.bias',
        'features.15.weight',
        'features.15.bias',
        'features.15.running_mean',
        'features.15.running_var',
        'features.17.weight',
        'features.17.bias',
        'features.18.weight',
        'features.18.bias',
        'features.18.running_mean',
        'features.18.running_var',
        'features.20.weight',
        'features.20.bias',
        'features.21.weight',
        'features.21.bias',
        'features.21.running_mean',
        'features.21.running_var',
        'features.23.weight',
        'features.23.bias',
        'features.24.weight',
        'features.24.bias',
        'features.24.running_mean',
        'features.24.running_var',
        'features.27.weight',
        'features.27.bias',
        'features.28.weight',
        'features.28.bias',
        'features.28.running_mean',
        'features.28.running_var',
        'features.30.weight',
        'features.30.bias',
        'features.31.weight',
        'features.31.bias',
        'features.31.running_mean',
        'features.31.running_var',
        'features.33.weight',
        'features.33.bias',
        'features.34.weight',
        'features.34.bias',
        'features.34.running_mean',
        'features.34.running_var',
        'features.36.weight',
        'features.36.bias',
        'features.37.weight',
        'features.37.bias',
        'features.37.running_mean',
        'features.37.running_var',
        'features.40.weight',
        'features.40.bias',
        'features.41.weight',
        'features.41.bias',
        'features.41.running_mean',
        'features.41.running_var',
        'features.43.weight',
        'features.43.bias',
        'features.44.weight',
        'features.44.bias',
        'features.44.running_mean',
        'features.44.running_var',
        'features.46.weight',
        'features.46.bias',
        'features.47.weight',
        'features.47.bias',
        'features.47.running_mean',
        'features.47.running_var',
        'features.49.weight',
        'features.49.bias',
        'features.50.weight',
        'features.50.bias',
        'features.50.running_mean',
        'features.50.running_var',
        'classifier.0.weight',
        'classifier.0.bias',
        'classifier.3.weight',
        'classifier.3.bias',
        'classifier.6.weight',
        'classifier.6.bias']


new_to_pretrain = {
        # stage 0
        'stage0_conv1.weight':
        'features.0.weight',
        'stage0_conv1.bias':
        'features.0.bias',
        'stage0_bn1.weight':
        'features.1.weight',
        'stage0_bn1.bias':
        'features.1.bias',
        'stage0_bn1.running_mean':
        'features.1.running_mean',
        'stage0_bn1.running_var':
        'features.1.running_var',
        'stage0_conv2.weight':
        'features.3.weight',
        'stage0_conv2.bias':
        'features.3.bias',
        'stage0_bn2.weight':
        'features.4.weight',
        'stage0_bn2.bias':
        'features.4.bias',
        'stage0_bn2.running_mean':
        'features.4.running_mean',
        'stage0_bn2.running_var':
        'features.4.running_var',
        # stage 1
        'stage1_conv1.weight':
        'features.7.weight',
        'stage1_conv1.bias':
        'features.7.bias',
        'stage1_bn1.weight':
        'features.8.weight',
        'stage1_bn1.bias':
        'features.8.bias',
        'stage1_bn1.running_mean':
        'features.8.running_mean',
        'stage1_bn1.running_var':
        'features.8.running_var',
        'stage1_conv2.weight':
        'features.10.weight',
        'stage1_conv2.bias':
        'features.10.bias',
        'stage1_bn2.weight':
        'features.11.weight',
        'stage1_bn2.bias':
        'features.11.bias',
        'stage1_bn2.running_mean':
        'features.11.running_mean',
        'stage1_bn2.running_var':
        'features.11.running_var',
        # stage 2
        'stage2_conv1.weight':
        'features.14.weight',
        'stage2_conv1.bias':
        'features.14.bias',
        'stage2_bn1.weight':
        'features.15.weight',
        'stage2_bn1.bias':
        'features.15.bias',
        'stage2_bn1.running_mean':
        'features.15.running_mean',
        'stage2_bn1.running_var':
        'features.15.running_var',
        'stage2_conv2.weight':
        'features.17.weight',
        'stage2_conv2.bias':
        'features.17.bias',
        'stage2_bn2.weight':
        'features.18.weight',
        'stage2_bn2.bias':
        'features.18.bias',
        'stage2_bn2.running_mean':
        'features.18.running_mean',
        'stage2_bn2.running_var':
        'features.18.running_var',
        'stage2_conv3.weight':
        'features.20.weight',
        'stage2_conv3.bias':
        'features.20.bias',
        'stage2_bn3.weight':
        'features.21.weight',
        'stage2_bn3.bias':
        'features.21.bias',
        'stage2_bn3.running_mean':
        'features.21.running_mean',
        'stage2_bn3.running_var':
        'features.21.running_var',
        'stage2_conv4.weight':
        'features.23.weight',
        'stage2_conv4.bias':
        'features.23.bias',
        'stage2_bn4.weight':
        'features.24.weight',
        'stage2_bn4.bias':
        'features.24.bias',
        'stage2_bn4.running_mean':
        'features.24.running_mean',
        'stage2_bn4.running_var':
        'features.24.running_var',
        # stage 3
        'stage3_conv1.weight':
        'features.27.weight',
        'stage3_conv1.bias':
        'features.27.bias',
        'stage3_bn1.weight':
        'features.28.weight',
        'stage3_bn1.bias':
        'features.28.bias',
        'stage3_bn1.running_mean':
        'features.28.running_mean',
        'stage3_bn1.running_var':
        'features.28.running_var',
        'stage3_conv2.weight':
        'features.30.weight',
        'stage3_conv2.bias':
        'features.30.bias',
        'stage3_bn2.weight':
        'features.31.weight',
        'stage3_bn2.bias':
        'features.31.bias',
        'stage3_bn2.running_mean':
        'features.31.running_mean',
        'stage3_bn2.running_var':
        'features.31.running_var',
        'stage3_conv3.weight':
        'features.33.weight',
        'stage3_conv3.bias':
        'features.33.bias',
        'stage3_bn3.weight':
        'features.34.weight',
        'stage3_bn3.bias':
        'features.34.bias',
        'stage3_bn3.running_mean':
        'features.34.running_mean',
        'stage3_bn3.running_var':
        'features.34.running_var',
        'stage3_conv4.weight':
        'features.36.weight',
        'stage3_conv4.bias':
        'features.36.bias',
        'stage3_bn4.weight':
        'features.37.weight',
        'stage3_bn4.bias':
        'features.37.bias',
        'stage3_bn4.running_mean':
        'features.37.running_mean',
        'stage3_bn4.running_var':
        'features.37.running_var',
        # stage 4
        'stage4_conv1.weight':
        'features.40.weight',
        'stage4_conv1.bias':
        'features.40.bias',
        'stage4_bn1.weight':
        'features.41.weight',
        'stage4_bn1.bias':
        'features.41.bias',
        'stage4_bn1.running_mean':
        'features.41.running_mean',
        'stage4_bn1.running_var':
        'features.41.running_var',
        'stage4_conv2.weight':
        'features.43.weight',
        'stage4_conv2.bias':
        'features.43.bias',
        'stage4_bn2.weight':
        'features.44.weight',
        'stage4_bn2.bias':
        'features.44.bias',
        'stage4_bn2.running_mean':
        'features.44.running_mean',
        'stage4_bn2.running_var':
        'features.44.running_var',
        'stage4_conv3.weight':
        'features.46.weight',
        'stage4_conv3.bias':
        'features.46.bias',
        'stage4_bn3.weight':
        'features.47.weight',
        'stage4_bn3.bias':
        'features.47.bias',
        'stage4_bn3.running_mean':
        'features.47.running_mean',
        'stage4_bn3.running_var':
        'features.47.running_var',
        'stage4_conv4.weight':
        'features.49.weight',
        'stage4_conv4.bias':
        'features.49.bias',
        'stage4_bn4.weight':
        'features.50.weight',
        'stage4_bn4.bias':
        'features.50.bias',
        'stage4_bn4.running_mean':
        'features.50.running_mean',
        'stage4_bn4.running_var':
        'features.50.running_var',
        }



class vgg_19_bn_features(nn.Module):

    def __init__(self):
        super(vgg_19_bn_features, self).__init__()
        # stage 0
        self.stage0_conv1 = nn.Conv2d(4, 64, kernel_size=3, padding=1)
        self.stage0_bn1 = nn.BatchNorm2d(64)
        self.stage0_relu1 = nn.ReLU(inplace=True)
        self.stage0_conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.stage0_bn2 = nn.BatchNorm2d(64)

        # stage 1
        self.stage1_relu0 = nn.ReLU(inplace=True)
        self.stage1_pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices = True)
        self.stage1_conv1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.stage1_bn1 = nn.BatchNorm2d(128)
        self.stage1_relu1 = nn.ReLU(inplace=True)
        self.stage1_conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.stage1_bn2 = nn.BatchNorm2d(128)

        # stage 2
        self.stage2_relu0 = nn.ReLU(inplace=True)
        self.stage2_pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices = True)
        self.stage2_conv1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.stage2_bn1 = nn.BatchNorm2d(256)
        self.stage2_relu1 = nn.ReLU(inplace=True)
        self.stage2_conv2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.stage2_bn2 = nn.BatchNorm2d(256)
        self.stage2_relu2 = nn.ReLU(inplace=True)
        self.stage2_conv3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.stage2_bn3 = nn.BatchNorm2d(256)
        self.stage2_relu3 = nn.ReLU(inplace=True)
        self.stage2_conv4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.stage2_bn4 = nn.BatchNorm2d(256)

        # stage 3
        self.stage3_relu0 = nn.ReLU(inplace=True)
        self.stage3_pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices = True)
        self.stage3_conv1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.stage3_bn1 = nn.BatchNorm2d(512)
        self.stage3_relu1 = nn.ReLU(inplace=True)
        self.stage3_conv2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.stage3_bn2 = nn.BatchNorm2d(512)
        self.stage3_relu2 = nn.ReLU(inplace=True)
        self.stage3_conv3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.stage3_bn3 = nn.BatchNorm2d(512)
        self.stage3_relu3 = nn.ReLU(inplace=True)
        self.stage3_conv4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.stage3_bn4 = nn.BatchNorm2d(512)

        # stage 4
        self.stage4_relu0 = nn.ReLU(inplace=True)
        self.stage4_pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices = True)
        self.stage4_conv1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.stage4_bn1 = nn.BatchNorm2d(512)
        self.stage4_relu1 = nn.ReLU(inplace=True)
        self.stage4_conv2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.stage4_bn2 = nn.BatchNorm2d(512)
        self.stage4_relu2 = nn.ReLU(inplace=True)
        self.stage4_conv3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.stage4_bn3 = nn.BatchNorm2d(512)
        self.stage4_relu3 = nn.ReLU(inplace=True)
        self.stage4_conv4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.stage4_bn4 = nn.BatchNorm2d(512)
        self.stage4_relu4 = nn.ReLU(inplace=True)

        self.load_pretrain()

    def load_pretrain(self, model_path = "./pretrained/vgg19_bn-c79401a0.pth"):
        model_dict = self.state_dict()
        pretrained_dict = torch.load(model_path)
        for name in model_dict:
            if name not in new_to_pretrain.keys():
                print(name)
                continue
            if name == "stage0_conv1.weight":
                model_dict[name][:, 0:3, :, :] = pretrained_dict[new_to_pretrain[name]]
                model_dict[name][:, 3:, :, :] = torch.tensor(0)
            else:
                assert model_dict[name].shape == pretrained_dict[new_to_pretrain[name]].shape
                model_dict[name] = pretrained_dict[new_to_pretrain[name]]

    def forward_stage0(self, x):
        x0 = self.stage0_conv1(x)
        x0 = self.stage0_bn1(x0)
        x0 = self.stage0_relu1(x0)
        x0 = self.stage0_conv2(x0)
        x0 = self.stage0_bn2(x0)
        return x0
    def forward_stage1(self, x0):
        x1, idx = self.stage1_pool(x0)
        x1 = self.stage1_relu0(x1)
        x1 = self.stage1_conv1(x1)
        x1 = self.stage1_bn1(x1)
        x1 = self.stage1_relu1(x1)
        x1 = self.stage1_conv2(x1)
        x1 = self.stage1_bn2(x1)
        return x1, idx
    def forward_stage2(self, x1):
        x2, idx = self.stage2_pool(x1)
        x2 = self.stage2_relu0(x2)
        x2 = self.stage2_conv1(x2)
        x2 = self.stage2_bn1(x2)
        x2 = self.stage2_relu1(x2)
        x2 = self.stage2_conv2(x2)
        x2 = self.stage2_bn2(x2)
        x2 = self.stage2_relu2(x2)
        x2 = self.stage2_conv3(x2)
        x2 = self.stage2_bn3(x2)
        x2 = self.stage2_relu3(x2)
        x2 = self.stage2_conv4(x2)
        x2 = self.stage2_bn4(x2)
        return x2, idx
    def forward_stage3(self, x2):
        x3, idx = self.stage3_pool(x2)
        x3 = self.stage3_relu0(x3)
        x3 = self.stage3_conv1(x3)
        x3 = self.stage3_bn1(x3)
        x3 = self.stage3_relu1(x3)
        x3 = self.stage3_conv2(x3)
        x3 = self.stage3_bn2(x3)
        x3 = self.stage3_relu2(x3)
        x3 = self.stage3_conv3(x3)
        x3 = self.stage3_bn3(x3)
        x3 = self.stage3_relu3(x3)
        x3 = self.stage3_conv4(x3)
        x3 = self.stage3_bn4(x3)
        return x3, idx

    def forward_stage4(self, x3):
        x4, idx = self.stage4_pool(x3)
        x4 = self.stage4_relu0(x4)
        x4 = self.stage4_conv1(x4)
        x4 = self.stage4_bn1(x4)
        x4 = self.stage4_relu1(x4)
        x4 = self.stage4_conv2(x4)
        x4 = self.stage4_bn2(x4)
        x4 = self.stage4_relu2(x4)
        x4 = self.stage4_conv3(x4)
        x4 = self.stage4_bn3(x4)
        x4 = self.stage4_relu3(x4)
        x4 = self.stage4_conv4(x4)
        x4 = self.stage4_bn4(x4)
        return x4, idx


    def forward(self, x, stage):
        if stage == 0:
            return self.forward_stage0(x)
        if stage == 1:
            return self.forward_stage1(x)
        if stage == 2:
            return self.forward_stage2(x)
        if stage == 3:
            return self.forward_stage3(x)
        if stage == 4:
            return self.forward_stage4(x)

        if stage == "all":
            x0 = self.forward_stage0(x)
            x1, _ = self.forward_stage1(x0)
            x2, _ = self.forward_stage2(x1)
            x3, _ = self.forward_stage3(x2)
            x4, _ = self.forward_stage4(x3)
            return x4



class decoder_module(nn.Module):

    def __init__(self, inp, oup):
        super(decoder_module, self).__init__()
        self.decoder = nn.Sequential(
                nn.ReLU(),
                nn.Conv2d(inp, oup, kernel_size=5, padding=2),
                )

    def forward(self, x, idx):
        y = self.decoder(x)
        y = F.max_unpool2d(y, idx, kernel_size = 2, stride = 2)
        return y

class theModel(nn.Module):

    def __init__(self):
        super(theModel, self).__init__()

        self.vgg = vgg_19_bn_features()

        self.stage5_relu0 = nn.ReLU(inplace=True)
        self.stage5_pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices = True)
        self.stage5_conv1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.stage5_bn1 = nn.BatchNorm2d(512)

        decoder_channels_5 = 512
        decoder_channels_4 = 512
        decoder_channels_3 = 512
        decoder_channels_2 = 256
        decoder_channels_1 = 128
        decoder_channels_0 = 64

        self.decoder_5 = decoder_module(512, decoder_channels_4)
        self.decoder_4 = decoder_module(512, decoder_channels_3)
        self.decoder_3 = decoder_module(512, decoder_channels_2)
        self.decoder_2 = decoder_module(256, decoder_channels_1)
        self.decoder_1 = decoder_module(128, decoder_channels_0)

        self.decoder_0 = nn.Sequential(
                nn.Conv2d(decoder_channels_0, 1, 3, 1, 1),
                )

    def forward(self, x):
        x0 = self.vgg(x, 0)
        x1, idx1 = self.vgg(x0, 1)
        x2, idx2 = self.vgg(x1, 2)
        x3, idx3 = self.vgg(x2, 3)
        x4, idx4 = self.vgg(x3, 4)
        x5 = self.stage5_relu0(x4)
        x5, idx5 = self.stage5_pool(x5)
        x5 = self.stage5_conv1(x5)
        x5 = self.stage5_bn1(x5)

        x4_decoder = self.decoder_5(x5, idx5)
        x3_decoder = self.decoder_4(x4_decoder, idx4)
        x2_decoder = self.decoder_3(x3_decoder, idx3)
        x1_decoder = self.decoder_2(x2_decoder, idx2)
        x0_decoder = self.decoder_1(x1_decoder, idx1)
        pred_alpha = self.decoder_0(x0_decoder)

        return pred_alpha

