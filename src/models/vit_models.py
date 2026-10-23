"""Classification baselines and CADLGI-Swin."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .build_vit_backbone import build_vit_sup_models, build_swin_model
from .mlp import MLP
from ..configs.config_node import CfgNode

def initialize_task_linears(module):
    """Match the task-layer initialization used by CADLGI-Rep and Swin."""
    for layer in module.modules():
        if isinstance(layer, nn.Linear):
            nn.init.trunc_normal_(layer.weight, std=0.02)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

class LabelSmoothingCrossEntropy(nn.Module):
    """Label smoothing cross entropy loss."""
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, x, target):
        confidence = 1. - self.smoothing
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

class ViT(nn.Module):
    """ViT-related model."""

    def __init__(self, cfg, load_pretrain=True, vis=False):
        super(ViT, self).__init__()

        # Only support end2end or linear transfer
        self.froze_enc = (cfg.MODEL.TRANSFER_TYPE == "linear")

        # For compatibility with trainer.py
        self.side = None

        self.build_backbone(cfg, load_pretrain, vis=vis)
        self.pretrained_info = getattr(self.enc, 'pretrained_info', None)
        self.cfg = cfg
        self.setup_head(cfg)
        self.head_dropout = nn.Dropout(cfg.MODEL.CADLGI.HEAD_DROP)

    def build_backbone(self, cfg, load_pretrain, vis):
        transfer_type = cfg.MODEL.TRANSFER_TYPE
        self.enc, self.feat_dim = build_vit_sup_models(
            cfg.DATA.FEATURE, cfg.DATA.CROPSIZE, None, cfg.MODEL.MODEL_ROOT,
            None, None, load_pretrain, vis
        )

        # Freeze encoder for linear probing
        if transfer_type == "linear":
            for k, p in self.enc.named_parameters():
                p.requires_grad = False

    def setup_head(self, cfg):
        self.head = MLP(
            input_dim=self.feat_dim,
            mlp_dims=[self.feat_dim] * self.cfg.MODEL.MLP_NUM + \
                [cfg.DATA.NUMBER_CLASSES], # noqa
            special_bias=True
        )

    def forward(self, x, return_feature=False):
        if self.froze_enc and self.enc.training:
            self.enc.eval()
        x = self.enc(x)  # batch_size x self.feat_dim

        if return_feature:
            return x, x
        x = self.head(self.head_dropout(x))
        return x

    @torch.no_grad()
    def embed(self, x):
        """Feature extraction."""
        if self.froze_enc and self.enc.training:
            self.enc.eval()
        return self.enc(x)

class Swin(ViT):
    """Swin-T classification baseline."""

    def setup_head(self, cfg):
        super().setup_head(cfg)
        initialize_task_linears(self.head)

    def build_backbone(self, cfg, load_pretrain, vis):
        self.enc, self.feat_dim = build_swin_model(
            cfg.DATA.FEATURE, cfg.DATA.CROPSIZE, None, cfg.MODEL.MODEL_ROOT,
            cross_layer=False)
        if cfg.MODEL.TRANSFER_TYPE == "linear":
            for parameter in self.enc.parameters():
                parameter.requires_grad = False


class CADLGISwin(nn.Module):
    """CADLGI-Swin: Swin-T features, CADLGI block, and classifier."""

    def __init__(self, cfg):
        super(CADLGISwin, self).__init__()

        self.cfg = cfg
        self.side = None  # For trainer compatibility
        self.supports_label_loss = True  # Tell trainer to pass labels

        # Build Swin backbone with cross_layer for multi-stage features
        self.enc, self.feat_dim = build_swin_model(
            cfg.DATA.FEATURE, cfg.DATA.CROPSIZE,
            None, cfg.MODEL.MODEL_ROOT,
            adapter_cfg=None, lora_cfg=None,
            cross_layer=True
        )
        self.pretrained_info = self.enc.pretrained_info

        cadlgi_cfg = getattr(cfg.MODEL, 'CADLGI', CfgNode())
        num_heads = getattr(cadlgi_cfg, 'NUM_HEADS', 4)
        use_algf = getattr(cadlgi_cfg, 'USE_ALGF', True)
        algf_version = getattr(cadlgi_cfg, 'ALGF_VERSION', 2)
        use_spatial_adaptive = getattr(cadlgi_cfg, 'USE_SPATIAL_ADAPTIVE', True)
        use_cadpa = getattr(cadlgi_cfg, 'USE_CADPA', True)
        max_prototypes = getattr(cadlgi_cfg, 'MAX_PROTOTYPES', 32)
        min_active_ratio = getattr(cadlgi_cfg, 'MIN_ACTIVE_RATIO', 0.25)
        label_smooth = getattr(cadlgi_cfg, 'LABEL_SMOOTH', 0.0)
        head_drop = getattr(cadlgi_cfg, 'HEAD_DROP', 0.2)

        # Feature size after Swin backbone (input_size / 32)
        self.feature_size = (cfg.DATA.CROPSIZE // 32, cfg.DATA.CROPSIZE // 32)

        from .modules.cadlgi import CADLGIBlock

        print(f"\nBuilding CADLGI-Swin:")
        print(f"  Backbone: {cfg.DATA.FEATURE}")
        print(f"  Feature dim: {self.feat_dim}")
        print(f"  Feature size: {self.feature_size}")

        self.cadlgi = CADLGIBlock(
            dim=self.feat_dim,
            query_size=self.feature_size,
            num_heads=num_heads,
            use_algf=use_algf,
            algf_version=algf_version,
            use_spatial_adaptive=use_spatial_adaptive,
            use_cadpa=use_cadpa,
            max_prototypes=max_prototypes,
            min_active_ratio=min_active_ratio,
        )

        # Classification head
        self.norm = nn.LayerNorm(self.feat_dim)
        self.head_drop_layer = nn.Dropout(head_drop)
        self.head = nn.Linear(self.feat_dim, cfg.DATA.NUMBER_CLASSES)
        # Initialize only new task layers; preserve every pretrained encoder tensor.
        initialize_task_linears(self.cadlgi)
        initialize_task_linears(self.head)

        # Loss function
        self.label_smooth = label_smooth
        if label_smooth > 0:
            self.ce = LabelSmoothingCrossEntropy(smoothing=label_smooth)
        else:
            self.ce = nn.CrossEntropyLoss()

        self._dynamic_stats = None

        print(f"CADLGI-Swin initialized.\n")

    def forward(self, x, label=None, return_feature=False):
        """Return logits, optional features, and optional classification loss."""
        # Swin backbone forward
        features = self.enc(x)

        if isinstance(features, (list, tuple)):
            feat = features[-1]  # [B, H*W, C] or [B, C, H, W]
        else:
            feat = features

        if feat.dim() == 4:  # [B, C, H, W]
            b, c, h, w = feat.shape
            feat = feat.flatten(2).transpose(1, 2)  # [B, H*W, C]

        # CADLGI forward
        if self.training:
            feat_out, stats = self.cadlgi(feat, return_dynamic_stats=True)
            self._dynamic_stats = stats
        else:
            feat_out = self.cadlgi(feat)

        # Global average pooling
        feat_pooled = feat_out.mean(dim=1)  # [B, C]

        # Classification head
        feat_pooled = self.norm(feat_pooled)
        feat_pooled = self.head_drop_layer(feat_pooled)
        logits = self.head(feat_pooled)

        if return_feature:
            return logits, feat_pooled

        # Training with loss
        if self.training and label is not None:
            loss = self.ce(logits, label)
            return logits, loss

        return logits

    @torch.no_grad()
    def embed(self, x):
        """Feature extraction."""
        features = self.enc(x)
        if isinstance(features, (list, tuple)):
            feat = features[-1]
        else:
            feat = features

        if feat.dim() == 4:
            b, c, h, w = feat.shape
            feat = feat.flatten(2).transpose(1, 2)

        feat_out = self.cadlgi(feat)
        return feat_out.mean(dim=1)

    def get_dynamic_stats(self):
        """Return cached prototype diagnostics."""
        return self._dynamic_stats
