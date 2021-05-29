# vovnet is from https://github.com/youngwanLEE/vovnet-detectron2 which is under Apache License 2.0

import pdb

from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F

def get_norm(norm, out_channels):
    norm = {
            "BN": nn.BatchNorm2d,
            "GN": lambda channels: nn.GroupNorm(32, channels),
            }[norm]
    return norm(out_channels)


_NORM = False

VoVNet19_eSE = {
    'stem': [64, 64, 128],
    "stage_conv_ch": [128, 160, 192, 224],
    "stage_out_ch": [256, 512, 768, 1024],
    "layer_per_block": 3,
    "block_per_stage": [1, 1, 1, 1],
    "eSE": True,
    "dw" : False
}

VoVNet39_eSE = {
    'stem': [64, 64, 128],
    "stage_conv_ch": [128, 160, 192, 224],
    "stage_out_ch": [256, 512, 768, 1024],
    "layer_per_block": 5,
    "block_per_stage": [1, 1, 2, 2],
    "eSE": True,
    "dw" : False
}

_STAGE_SPECS = {
    "V-19-eSE": VoVNet19_eSE,
    "V-39-eSE": VoVNet39_eSE,
}

def dw_conv3x3(in_channels, out_channels, module_name, postfix,
            stride=1, kernel_size=3, padding=1):
    """3x3 convolution with padding"""
    return [
        ('{}_{}/dw_conv3x3'.format(module_name, postfix),
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=kernel_size,
                      stride=stride,
                      padding=padding,
                      groups=out_channels,
                      bias=False)),
        ('{}_{}/pw_conv1x1'.format(module_name, postfix),
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      groups=1,
                      bias=False)),
        ('{}_{}/pw_norm'.format(module_name, postfix), get_norm(_NORM, out_channels)),
        ('{}_{}/pw_relu'.format(module_name, postfix), nn.ReLU(inplace=True)),
    ]

def conv3x3(
    in_channels, out_channels, module_name, postfix, stride=1, groups=1, kernel_size=3, padding=1
):
    """3x3 convolution with padding"""
    return [
        (
            f"{module_name}_{postfix}/conv",
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False,
            ),
        ),
        (f"{module_name}_{postfix}/norm", get_norm(_NORM, out_channels)),
        (f"{module_name}_{postfix}/relu", nn.ReLU(inplace=True)),
    ]


def conv1x1(
    in_channels, out_channels, module_name, postfix, stride=1, groups=1, kernel_size=1, padding=0
):
    """1x1 convolution with padding"""
    return [
        (
            f"{module_name}_{postfix}/conv",
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False,
            ),
        ),
        (f"{module_name}_{postfix}/norm", get_norm(_NORM, out_channels)),
        (f"{module_name}_{postfix}/relu", nn.ReLU(inplace=True)),
    ]


class Hsigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(Hsigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return F.relu6(x + 3.0, inplace=self.inplace) / 6.0


class eSEModule(nn.Module):
    def __init__(self, channel, reduction=4):
        super(eSEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channel, channel, kernel_size=1, padding=0)
        self.hsigmoid = Hsigmoid()

    def forward(self, x):
        input = x
        x = self.avg_pool(x)
        x = self.fc(x)
        x = self.hsigmoid(x)
        return input * x


class _OSA_module(nn.Module):
    def __init__(
        self, in_ch, stage_ch, concat_ch, layer_per_block, module_name, SE=False, identity=False, depthwise=False
    ):

        super(_OSA_module, self).__init__()

        self.identity = identity
        self.depthwise = depthwise
        self.isReduced = False
        self.layers = nn.ModuleList()
        in_channel = in_ch
        for i in range(layer_per_block):
            self.layers.append(
                nn.Sequential(OrderedDict(conv3x3(in_channel, stage_ch, module_name, i)))
            )
            in_channel = stage_ch

        # feature aggregation
        in_channel = in_ch + layer_per_block * stage_ch
        self.concat = nn.Sequential(
            OrderedDict(conv1x1(in_channel, concat_ch, module_name, "concat"))
        )

        self.ese = eSEModule(concat_ch)

    def forward(self, x):

        identity_feat = x

        output = []
        output.append(x)
        for layer in self.layers:
            x = layer(x)
            output.append(x)

        x = torch.cat(output, dim=1)
        xt = self.concat(x)

        xt = self.ese(xt)

        if self.identity:
            xt = xt + identity_feat

        return xt


class _OSA_stage(nn.Sequential):
    def __init__(
        self,
        in_ch,
        stage_ch,
        concat_ch,
        block_per_stage,
        layer_per_block,
        stage_num, SE=False,
        depthwise=False):

        super(_OSA_stage, self).__init__()

        if not stage_num == 2:
            self.add_module("Pooling", nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True))

        if block_per_stage != 1:
            SE = False
        module_name = f"OSA{stage_num}_1"
        self.add_module(
            module_name, _OSA_module(in_ch, stage_ch, concat_ch, layer_per_block, module_name, SE, depthwise=depthwise)
        )
        for i in range(block_per_stage - 1):
            if i != block_per_stage - 2:  # last block
                SE = False
            module_name = f"OSA{stage_num}_{i + 2}"
            self.add_module(
                module_name,
                _OSA_module(
                    concat_ch, stage_ch, concat_ch, layer_per_block, module_name, SE, identity=True, depthwise=depthwise
                ),
            )




class theStem(nn.Module):
    def __init__(self, input_ch, stem_ch):
        super(theStem, self).__init__()
        stem = conv3x3(input_ch, stem_ch[0], "stem", "1", 2)
        stem += conv3x3(stem_ch[0], stem_ch[1], "stem", "2", 1)
        stem += conv3x3(stem_ch[1], stem_ch[2], "stem", "3", 2)

        for name, mod in stem:
            self.add_module(name, mod)
    def forward(self, x):

        x = getattr(self, "stem_1/conv")(x)
        x = getattr(self, "stem_1/norm")(x)
        x = getattr(self, "stem_1/relu")(x)

        x = getattr(self, "stem_2/conv")(x)
        x = getattr(self, "stem_2/norm")(x)
        x_2 = x.clone()
        x = getattr(self, "stem_2/relu")(x)

        x = getattr(self, "stem_3/conv")(x)
        x = getattr(self, "stem_3/norm")(x)
        x = getattr(self, "stem_3/relu")(x)
        return x_2, x


class VoVNet(nn.Module):
    def __init__(self, input_ch, myArch):
        """
        Args:
            input_ch(int) : the number of input channel
            out_features (list[str]): name of the layers whose outputs should
                be returned in forward. Can be anything in "stem", "stage2" ...
        """
        super(VoVNet, self).__init__()

        global _NORM
        _NORM = "BN"


        if myArch == "V-19-eSE":
            stage_specs = _STAGE_SPECS["V-19-eSE"]

        if myArch == "V-39-eSE":
            stage_specs = _STAGE_SPECS["V-39-eSE"]


        stem_ch = stage_specs["stem"]
        # [64, 64, 128]

        config_stage_ch = stage_specs["stage_conv_ch"]
        # [128, 160, 192, 224]

        config_concat_ch = stage_specs["stage_out_ch"]
        # [256, 512, 768, 1024]

        block_per_stage = stage_specs["block_per_stage"]
        # [1, 1, 2, 2]

        layer_per_block = stage_specs["layer_per_block"]
        # 5

        SE = stage_specs["eSE"]
        # True

        depthwise = stage_specs["dw"]
        # False

        self._out_features = ["stage1", "stage2", "stage3", "stage4", "stage5"]

        # Stem module
        self.stem = theStem(input_ch, stem_ch)

        current_stirde = 4
        self._out_feature_strides = {"stage1": 2, "stage2": current_stirde}
        self._out_feature_channels = {"stage1": stem_ch[1]}

        stem_out_ch = [stem_ch[2]]
        # [128]

        in_ch_list = stem_out_ch + config_concat_ch[:-1]
        # [128, 256, 512, 768]

        # OSA stages
        self.stage_names = []
        for i in range(4):  # num_stages
            name = "stage%d" % (i + 2)  # stage 2 ... stage 5
            self.stage_names.append(name)
            self.add_module(
                name,
                _OSA_stage(
                    in_ch_list[i],
                    config_stage_ch[i],
                    config_concat_ch[i],
                    block_per_stage[i],
                    layer_per_block,
                    i + 2,
                    SE,
                    depthwise,
                ),
            )

            self._out_feature_channels[name] = config_concat_ch[i]
            if not i == 0:
                self._out_feature_strides[name] = current_stirde = int(current_stirde * 2)

        # initialize weights
        #self._initialize_weights()
        # Optionally freeze (requires_grad=False) parts of the backbone
        #self._freeze_backbone(-1)

        # load pretrained model
        ifPretrain = True
        if ifPretrain:
            if myArch == "V-19-eSE":
                pretrained_dict = torch.load("./pretrained/vovnet19_ese_detectron2.pth")
            if myArch == "V-39-eSE":
                pretrained_dict = torch.load("./pretrained/vovnet39_ese_detectron2.pth")
            model_dict = self.state_dict()
            for name in pretrained_dict:
                tmp_name = name
                name = name[19:]
                if name not in model_dict:
                    print(name, " not in model_dict!!!!!!!!!")
                    continue
                if name == "stem.stem_1/conv.weight":
                    model_dict[name][:, :3, :, :] = pretrained_dict[tmp_name][:,[2,1,0],:,:]
                    model_dict[name][:, 3:, :, :] = 0
                    continue
                model_dict[name] = pretrained_dict[tmp_name]

            self.load_state_dict(model_dict)


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)

    def _freeze_backbone(self, freeze_at):
        if freeze_at < 0:
            return

        for stage_index in range(freeze_at):
            if stage_index == 0:
                m = self.stem  # stage 0 is the stem
            else:
                m = getattr(self, "stage" + str(stage_index + 1))
            for p in m.parameters():
                p.requires_grad = False
                FrozenBatchNorm2d.convert_frozen_batchnorm(self)

    def forward(self, x):
        outputs = {}
        stage1, x = self.stem(x)

        outputs["stage1"] = stage1
        for name in self.stage_names:
            x = getattr(self, name)(x)
            outputs[name] = x

        return outputs

class preprocess_trimap(nn.Module):
    def __init__(self):
        super(preprocess_trimap, self).__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(2, 8, kernel_size = 7, stride = 2, padding=3, bias = False)
        self.norm1 = nn.BatchNorm2d(8)
        self.prelu1 = nn.PReLU(8)
        self.conv2 = nn.Conv2d(8, 16, kernel_size = 7, stride = 2, padding=3, bias = False)
        self.norm2 = nn.BatchNorm2d(16)
        self.prelu2 = nn.PReLU(16)
        self.conv3 = nn.Conv2d(16, 32, kernel_size = 7, stride = 2, padding=3, bias = False)
        self.norm3 = nn.BatchNorm2d(32)
        self.prelu3 = nn.PReLU(32)

    def forward(self, x):

        trimap = x[:, 3,:,:]
        trimap = trimap[:, None, :, :]
        inner0 = self.relu(trimap-3./4.) * 4
        outer0 = 1 - self.relu(1./4. - trimap) * 4
        #trimaps = torch.cat([x[:,3,:,:], inner0, outer0], dim = 1)
        trimaps = torch.cat([inner0, outer0], dim = 1)
        trimaps = self.conv1(trimaps)
        trimaps = self.norm1(trimaps)
        trimaps = self.prelu1(trimaps)
        trimaps = self.conv2(trimaps)
        trimaps = self.norm2(trimaps)
        trimaps = self.prelu2(trimaps)
        trimaps = self.conv3(trimaps)
        trimaps = self.norm3(trimaps)
        trimaps = self.prelu3(trimaps)
        #inner0 = F.avg_pool2d(inner0, (8, 8), stride = 8)
        #outer0 = F.avg_pool2d(outer0, (8, 8), stride = 8)
        return trimaps

class trimap_fusion(nn.Module):
    def __init__(self):
        super(trimap_fusion, self).__init__()

        channel = 32 + 64
        for i in range(1, 6):
            self.add_module("conv" + str(i), nn.Conv2d( channel, 64, 5, 1, 2, bias = False))
            self.add_module("norm" + str(i), nn.BatchNorm2d(64))
            self.add_module("prelu" + str(i), nn.PReLU(64))
            channel = 64

        self.conv_concat = nn.Conv2d( 64*6, 64, 1, 1, 0, bias = False)
        self.norm_concat = nn.BatchNorm2d(64)
        self.prelu_concat = nn.PReLU(64)

    def forward(self, trimaps, feature):
        output = []
        output.append(feature)

        trimaps = torch.cat([trimaps, feature], dim = 1)
        for i in range(1, 6):
            trimaps = getattr(self, "conv" + str(i))(trimaps)
            trimaps = getattr(self, "norm" + str(i))(trimaps)
            trimaps = getattr(self, "prelu" + str(i))(trimaps)
            output.append(trimaps)

        x = torch.cat(output, dim = 1)
        xt = self.conv_concat(x)
        xt = self.norm_concat(xt)
        xt = self.prelu_concat(xt)

        return xt


class skipModule(nn.Module):
    def __init__(self, inChannels):
        super(skipModule, self).__init__()

        self.skip_5 = nn.Sequential(
                OrderedDict([
                    ("conv1", nn.Conv2d(inChannels['stage5'], 64, 3, 1, 1, bias = False)),
                    ("norm1", nn.BatchNorm2d(64)),
                ])
                )

        self.skip_4 = nn.Sequential(
                OrderedDict([
                    ("conv1", nn.Conv2d(inChannels['stage4'], 64, 3, 1, 1, bias = False)),
                    ("norm1", nn.BatchNorm2d(64)),
                ])
                )

        self.skip_3 = nn.Sequential(
                OrderedDict([
                    ("conv1", nn.Conv2d(inChannels['stage3'], 64, 3, 1, 1, bias = False)),
                    ("norm1", nn.BatchNorm2d(64)),
                ])
                )

        self.detail_stem0 = nn.Sequential(
                OrderedDict([
                    ("conv1", nn.Conv2d(3, 32, 3, 1, 1, bias = False)),
                    ("norm1", nn.BatchNorm2d(32)),
                ])
                )
        self.detail_stem1 = nn.Sequential(
                OrderedDict([
                    ("prelu1", nn.PReLU(32)),
                    ("conv1", nn.Conv2d(32, 64, 3, 2, 1, bias = False)),
                    ("norm1", nn.BatchNorm2d(64)),
                    ("prelu2", nn.PReLU(64)),
                    ("conv2", nn.Conv2d(64, 64, 3, 1, 1, bias = False)),
                    ("norm2", nn.BatchNorm2d(64)),
                ])
                )
        self.detail_stem2 = nn.Sequential(
                OrderedDict([
                    ("prelu1", nn.PReLU(64)),
                    ("conv1", nn.Conv2d(64, 64, 3, 2, 1, bias = False)),
                    ("norm1", nn.BatchNorm2d(64)),
                    ("prelu2", nn.PReLU(64)),
                    ("conv2", nn.Conv2d(64, 64, 3, 1, 1, bias = False)),
                    ("norm2", nn.BatchNorm2d(64)),
                ])
                )

    def forward(self, image, inFeatures):
        out = {}
        out['stage5'] = self.skip_5 (inFeatures['stage5'])
        out['stage4'] = self.skip_4 (inFeatures['stage4'])
        out['stage3'] = self.skip_3 (inFeatures['stage3'])

        out['stage0'] = self.detail_stem0(image)
        out['stage1'] = self.detail_stem1(out['stage0'])
        out['stage2'] = self.detail_stem2(out['stage1'])
        return out

class decoder_module(nn.Module):

    def __init__(self, inp, oup):
        super(decoder_module, self).__init__()


        self.decoder = nn.Sequential(
                OrderedDict([
                    ("prelu", nn.PReLU(inp)),
                    ("upsampe", nn.UpsamplingNearest2d(scale_factor = 2)),
                    ("conv", nn.Conv2d(inp, oup, 3, 1, 1, bias = False)),
                    ("norm", nn.BatchNorm2d(oup)),
                ])
                )

    def forward(self, lower):
        return self.decoder(lower)



class theModel(nn.Module):

    def __init__(self, myArch = "V-19-eSE"):
        super(theModel, self).__init__()

        self.backbone = VoVNet(4, myArch = myArch)
        self.skip = skipModule(self.backbone._out_feature_channels)

        decoder_channels_5 = 64
        decoder_channels_4 = 64
        decoder_channels_3 = 64
        decoder_channels_2 = 64
        decoder_channels_1 = 64
        decoder_channels_0 = 32

        self.decoder_5 = decoder_module(64, 64)
        self.decoder_4 = decoder_module(64, 64)
        self.decoder_3 = decoder_module(64, 64)
        self.decoder_2 = decoder_module(64, 64)
        self.decoder_1 =nn.Sequential(
                        OrderedDict([
                            ("prelu", nn.PReLU(64)),
                            ("upsampe", nn.UpsamplingNearest2d(scale_factor = 2)),
                            ("conv", nn.Conv2d(64, 32, 3, 1, 1, bias = False)),
                            ("norm", nn.BatchNorm2d(32)),
                        ])
                        )

        self.decoder_0 = nn.Sequential(
                OrderedDict([
                    ("prelu1", nn.PReLU(32)),
                    ("conv1", nn.Conv2d(32, 32, 3, 1, 1, bias = False)),
                    ("norm1", nn.BatchNorm2d(32)),
                    ("prelu2", nn.PReLU(32)),
                    ("conv2", nn.Conv2d(32, 1, 1, 1, 0)),
                ])
                )

    def forward(self, x):

        encoder_out = self.backbone(x)
        skip_out = self.skip(x[:,:3,:,:], encoder_out)

        skip_out['stage4'] = skip_out['stage4'] + self.decoder_5(skip_out['stage5'])
        skip_out['stage3'] = skip_out['stage3'] + self.decoder_4(skip_out['stage4'])
        skip_out['stage2'] = skip_out['stage2'] + self.decoder_3(skip_out['stage3'])
        skip_out['stage1'] = skip_out['stage1'] + self.decoder_2(skip_out['stage2'])
        skip0 = skip_out['stage0'] + self.decoder_1(skip_out['stage1'])
        pred_alpha = self.decoder_0(skip0)

        return pred_alpha

if __name__ == "__main__":

    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '6'
    model = VoVNet(3)
    model.eval()
    model.cuda()
    dump_x = torch.randn(1, 4, 2048, 2048).cuda()
    y = model(dump_x)
    pdb.set_trace()



