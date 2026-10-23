"""ConvNeXt classification baseline with end-to-end training support."""
import torch
import torch.nn as nn
from collections import OrderedDict
from .mlp import MLP
from .convnext_backbone.convnext import convnext_tiny
from .resnet import ResNet
FEAT2DIM = {"tiny": 768}

class ConvNeXt(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.side = None
        if cfg.DATA.FEATURE != "imagenet_sup_rnx_tiny":
            raise ValueError("Only manuscript ConvNeXt-T is retained")
        model = convnext_tiny(pretrained=bool(cfg.MODEL.MODEL_ROOT))
        self.norm = model.norm
        self.frozen_layers = nn.Identity()
        self.tuned_layers = nn.Sequential(OrderedDict([
            ("downsample_layer1", model.downsample_layers[0]),
            ("stage1", model.stages[0]),
            ("downsample_layer2", model.downsample_layers[1]),
            ("stage2", model.stages[1]),
            ("downsample_layer3", model.downsample_layers[2]),
            ("stage3", model.stages[2]),
            ("downsample_layer4", model.downsample_layers[3]),
            ("stage4", model.stages[3])
        ]))
        if cfg.MODEL.TRANSFER_TYPE == "linear":
            self.frozen_layers, self.tuned_layers = self.tuned_layers, nn.Identity()
            for parameter in self.frozen_layers.parameters():
                parameter.requires_grad = False
        self.setup_head(cfg)

    def get_outputdim(self):
        backbone_arch = self.cfg.DATA.FEATURE.split("_")[-1]
        return FEAT2DIM[backbone_arch]

    def setup_head(self, cfg):
        self.head = MLP(input_dim=self.get_outputdim(),
                        mlp_dims=[self.get_outputdim()] * cfg.MODEL.MLP_NUM + [cfg.DATA.NUMBER_CLASSES],
                        special_bias=True)


    def get_features(self, x):
        self.frozen_layers.eval()
        x = self.tuned_layers(self.frozen_layers(x))
        return self.norm(x.mean([-2, -1]))

    def forward(self, x, return_feature=False):
        x = self.get_features(x)
        return x if return_feature else self.head(x)
