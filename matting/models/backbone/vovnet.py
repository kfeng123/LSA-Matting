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
    def __init__(self, input_ch, myArch, lastStage = 5):
        """
        Args:
            input_ch(int) : the number of input channel
            out_features (list[str]): name of the layers whose outputs should
                be returned in forward. Can be anything in "stem", "stage2" ...
        """
        super(VoVNet, self).__init__()

        global _NORM
        _NORM = "BN"

        self.lastStage = lastStage


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

        for i in range(self.lastStage + 1 - 2):  # num_stages
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

if __name__ == "__main__":

    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '6'
    model = VoVNet(3)
    model.eval()
    model.cuda()
    dump_x = torch.randn(1, 4, 2048, 2048).cuda()
    y = model(dump_x)
    pdb.set_trace()

