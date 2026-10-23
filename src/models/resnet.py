"""ResNet classification baseline with end-to-end training support."""
import torch
import torch.nn as nn
from collections import OrderedDict
from .mlp import MLP
from torchvision import models

class ResNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.side = None
        if cfg.DATA.FEATURE != "imagenet_sup_rn50":
            raise ValueError("Only manuscript ResNet50 is retained")
        model = models.resnet50(pretrained=bool(cfg.MODEL.MODEL_ROOT))
        self.frozen_layers = nn.Identity()
        self.tuned_layers = nn.Sequential(OrderedDict([
            ("conv1", model.conv1),
            ("bn1", model.bn1),
            ("relu", model.relu),
            ("maxpool", model.maxpool),
            ("layer1", model.layer1),
            ("layer2", model.layer2),
            ("layer3", model.layer3),
            ("layer4", model.layer4),
            ("avgpool", model.avgpool)
        ]))
        if cfg.MODEL.TRANSFER_TYPE == "linear":
            self.frozen_layers, self.tuned_layers = self.tuned_layers, nn.Identity()
            for parameter in self.frozen_layers.parameters():
                parameter.requires_grad = False
        self.setup_head(cfg)

    def get_outputdim(self):
        if self.cfg.DATA.FEATURE == "imagenet_sup_rn34" or self.cfg.DATA.FEATURE == "imagenet_sup_rn18":
            out_dim = 512
        else:
            out_dim = 2048
        return out_dim

    def setup_head(self, cfg):
        out_dim = self.get_outputdim()
        self.head = MLP(
            input_dim=out_dim,
            mlp_dims=[out_dim] * self.cfg.MODEL.MLP_NUM + \
                [cfg.DATA.NUMBER_CLASSES],
            special_bias=True
        )

    def get_features(self, x):
        self.frozen_layers.eval()
        return self.tuned_layers(self.frozen_layers(x)).flatten(1)

    def forward(self, x, return_feature=False):
        x = self.get_features(x)
        return x if return_feature else self.head(x)
