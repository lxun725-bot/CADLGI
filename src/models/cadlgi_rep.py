"""CADLGI-Rep and the unmodified RepViT classification baseline."""
import os
from typing import Optional, Tuple, List, Union
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from timm.loss import LabelSmoothingCrossEntropy
from .backbones.repvit import RepViT, repvit_m0_9, repvit_m1_0, repvit_m1_1, repvit_m1_5, repvit_m2_3
from .modules.cadlgi import CADLGIBlock
from .backbones.checkpoint import load_repvit_pretrained

REPVIT_PRETRAINED_URLS = {
    'repvit_m0_9': 'https://github.com/THU-MIG/RepViT/releases/download/v1.0/repvit_m0_9_distill_300e.pth',
    'repvit_m1_0': 'https://github.com/THU-MIG/RepViT/releases/download/v1.0/repvit_m1_0_distill_300e.pth',
    'repvit_m1_1': 'https://github.com/THU-MIG/RepViT/releases/download/v1.0/repvit_m1_1_distill_300e.pth',
    'repvit_m1_5': 'https://github.com/THU-MIG/RepViT/releases/download/v1.0/repvit_m1_5_distill_300e.pth',
    'repvit_m2_3': 'https://github.com/THU-MIG/RepViT/releases/download/v1.0/repvit_m2_3_distill_300e.pth',
}

class RepViTBackbone(nn.Module):
    """
    Wrapper for official RepViT to support multi-stage feature output for CADLGI.
    """
    def __init__(
        self,
        repvit_model: RepViT,
        cross_layer: bool = True
    ):
        super().__init__()
        self.backbone = repvit_model
        self.cross_layer = cross_layer

        # Remove classifier if exists
        if hasattr(self.backbone, 'classifier'):
            self.backbone.classifier = nn.Identity()

        # Get stage boundaries from config
        self.stage_boundaries = self._get_stage_boundaries()
        self.stage_out_channels = self._get_stage_channels()

    def _get_stage_boundaries(self) -> List[int]:
        """Get stage boundaries from RepViT config."""
        cfgs = self.backbone.cfgs
        boundaries = []
        for i, cfg in enumerate(cfgs):
            k, t, c, use_se, use_hs, s = cfg
            if s == 2:  # Downsample
                # features[0] is the stem; cfg[i] is features[i+1].
                # End the preceding stage BEFORE the downsampling block.
                boundaries.append(i)
        boundaries.append(len(cfgs))  # Last executable feature, not len(cfgs)+1
        return boundaries

    def _get_stage_channels(self) -> List[int]:
        """Get output channels for each stage."""
        cfgs = self.backbone.cfgs
        return [cfgs[index - 1][2] for index in self.stage_boundaries]

    def forward(self, x: Tensor) -> Union[Tensor, List[Tensor]]:
        """Forward pass with multi-stage output."""
        if not self.cross_layer:
            # Standard forward
            for f in self.backbone.features:
                x = f(x)
            return x

        # Multi-stage output for CADLGI
        features = []
        x = self.backbone.features[0](x)  # Patch embed

        stage_idx = 0
        for i, layer in enumerate(self.backbone.features[1:], start=1):
            x = layer(x)

            # Check if we hit a stage boundary
            if i in self.stage_boundaries:
                # Convert to [B, N, C] format
                b, c, h, w = x.shape
                feat = x.flatten(2).transpose(1, 2)  # [B, H*W, C]
                features.append(feat)
                stage_idx += 1

        return features

class CADLGIRep(nn.Module):
    """CADLGI-Rep: RepViT features, CADLGI block, and classifier."""
    def __init__(
        self,
        backbone_type: str = 'repvit_m1_0',
        input_size: int = 224,
        num_classes: int = 6,
        num_heads: int = 4,
        use_algf: bool = True,
        algf_version: int = 2,                 # 2: ALGF; 1: static fusion control.
        use_spatial_adaptive: bool = True,    # Spatial gating is available in ALGF v2.
        use_cadpa: bool = True,
        max_prototypes: int = 32,
        min_active_ratio: float = 0.25,
        label_smooth: float = 0.0,
        head_drop: float = 0.2,
        pretrained_path: Optional[str] = None,
    ):
        super().__init__()

        self.backbone_type = backbone_type
        self.input_size = input_size
        self.num_classes = num_classes
        self.use_cadpa = use_cadpa
        self.algf_version = algf_version

        # Build official RepViT backbone
        REPVIT_MODELS = {
            'repvit_m0_9': repvit_m0_9,
            'repvit_m1_0': repvit_m1_0,
            'repvit_m1_1': repvit_m1_1,
            'repvit_m1_5': repvit_m1_5,
            'repvit_m2_3': repvit_m2_3,
        }

        assert backbone_type in REPVIT_MODELS, f"Unknown backbone: {backbone_type}"
        repvit_model = REPVIT_MODELS[backbone_type](
            pretrained=False,
            num_classes=0,  # No classifier
            distillation=False
        )

        # Wrap with CADLGI-compatible backbone
        self.backbone = RepViTBackbone(repvit_model, cross_layer=True)

        # Get feature dimension from last stage
        self.dim = self.backbone.stage_out_channels[-1]

        # Feature size after backbone (input_size / 32)
        self.feature_size = (input_size // 32, input_size // 32)

        # CADLGI attention module with dynamic prototype support
        self.cadlgi = CADLGIBlock(
            dim=self.dim,
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
        self.norm = nn.LayerNorm(self.dim)
        self.head_drop = nn.Dropout(head_drop)
        self.head = nn.Linear(self.dim, num_classes)

        # Loss function
        self.label_smooth = label_smooth
        if label_smooth > 0:
            self.ce = LabelSmoothingCrossEntropy(smoothing=label_smooth)
        else:
            self.ce = nn.CrossEntropyLoss()

        self.pretrained_info = {'mode': 'scratch', 'backbone': backbone_type}

        # Load pretrained weights
        # None selects the default pretrained source.
        if pretrained_path is None:
            pretrained_path = "auto"

        if pretrained_path:
            self._load_pretrained(pretrained_path)

        # For trainer compatibility
        self.side = None
        self.supports_label_loss = True  # Tell trainer to pass labels
        self._enable_interpretability = False

        # Cached prototype diagnostics.
        self._dynamic_stats = None

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights for CADLGI attention and head."""
        from timm.models.vision_transformer import trunc_normal_

        for m in self.cadlgi.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        trunc_normal_(self.head.weight, std=0.02)
        if self.head.bias is not None:
            nn.init.zeros_(self.head.bias)

    def _load_pretrained(self, path: str):
        self.pretrained_info = load_repvit_pretrained(
            self.backbone.backbone, path, self.backbone_type, REPVIT_PRETRAINED_URLS)

    def forward(self, x: torch.Tensor, label: Optional[torch.Tensor] = None):
        """Forward pass.

        Args:
            x: Input images [B, 3, H, W]
            label: Optional labels [B] for loss computation

        Returns:
            - If training and label provided: (logits, loss)
            - Otherwise: logits only
        """
        # Extract multi-stage features from backbone
        features = self.backbone(x)  # List of [B, N, C] for each stage

        # Use last stage features
        feat = features[-1]  # [B, N, C]

        # Apply CADLGI attention with optional dynamic prototype stats
        if self.use_cadpa:
            cadlgi_output = self.cadlgi(feat, return_dynamic_stats=True)
            if isinstance(cadlgi_output, tuple):
                feat, dynamic_stats = cadlgi_output
                self._dynamic_stats = dynamic_stats
            else:
                feat = cadlgi_output
        else:
            feat = self.cadlgi(feat)

        # Shared CADLGI classifier order: GAP -> LayerNorm -> Dropout -> Linear.
        feat = self.head_drop(self.norm(feat.mean(dim=1)))
        logits = self.head(feat)
        if self.training and label is not None:
            return logits, self.ce(logits, label)

        # Eval mode: always return logits only (even if label provided)
        return logits

    def forward_with_interpretability(self, x: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """Forward pass with interpretability outputs."""
        logits = self.forward(x)
        if isinstance(logits, tuple):
            logits = logits[0]
        return logits, {'dynamic_stats': self.cadlgi.get_dynamic_stats(),
                        'feature_size': self.feature_size}

    def fuse_for_inference(self):
        raise NotImplementedError('Conv-BN fusion is not implemented or validated; use the unfused model.')

    def get_params_count(self) -> dict:
        """Get parameter counts."""
        backbone_params = sum(p.numel() for p in self.backbone.parameters())
        cadlgi_params = sum(p.numel() for p in self.cadlgi.parameters())
        head_params = sum(p.numel() for p in self.head.parameters()) + \
                     sum(p.numel() for p in self.norm.parameters())
        total_params = sum(p.numel() for p in self.parameters())

        dynamic_params = 0
        if self.use_cadpa and hasattr(self.cadlgi, 'cadpa'):
            dynamic_params += sum(p.numel() for p in self.cadlgi.cadpa.cce.parameters())
            dynamic_params += sum(p.numel() for p in self.cadlgi.cadpa.dpg.parameters())
            dynamic_params += sum(p.numel() for p in self.cadlgi.cadpa.prototype_generator.parameters())

        return {
            'backbone': backbone_params,
            'cadlgi_attention': cadlgi_params,
            'dynamic_prototype': dynamic_params,
            'head': head_params,
            'total': total_params,
            'total_M': total_params / 1e6,
        }


    def get_dynamic_prototype_stats(self) -> dict:
        """Return cached prototype activation statistics."""
        if not self.use_cadpa:
            return {'error': 'Dynamic prototype selection not enabled'}

        if self._dynamic_stats is None:
            return {'error': 'No cached stats. Run forward pass first.'}

        return self._dynamic_stats

    def analyze_prototype_activation(self, images: torch.Tensor) -> dict:
        """Run inference and return prototype diagnostics."""
        if not self.use_cadpa:
            return {'error': 'Dynamic prototype not enabled'}

        with torch.no_grad():
            _ = self.forward(images)

        gate_stats = self.cadlgi.cadpa.dpg.get_stats()

        return {
            'num_active_per_sample': gate_stats.get('num_active'),
            'importance_scores': gate_stats.get('importance_scores'),
            'gate_mask': gate_stats.get('gate_mask'),
            'dynamic_stats': self._dynamic_stats,
        }

def build_cadlgi_rep(cfg) -> CADLGIRep:
    """Build CADLGI-Rep from the model configuration."""
    cadlgi_cfg = getattr(cfg.MODEL, 'CADLGI', None)

    if cadlgi_cfg is None:
        # Default config
        backbone_type = 'repvit_m1_0'
        num_heads = 4
        use_algf = True
        algf_version = 2
        use_spatial_adaptive = True
        use_cadpa = True
        max_prototypes = 32
        min_active_ratio = 0.25
        label_smooth = 0.0
        head_drop = 0.2
        use_pretrained = True
        pretrained_path = None
    else:
        backbone_type = getattr(cadlgi_cfg, 'BACKBONE_TYPE', 'repvit_m1_0')
        num_heads = getattr(cadlgi_cfg, 'NUM_HEADS', 4)
        use_algf = getattr(cadlgi_cfg, 'USE_ALGF', True)
        algf_version = getattr(cadlgi_cfg, 'ALGF_VERSION', 2)
        use_spatial_adaptive = getattr(cadlgi_cfg, 'USE_SPATIAL_ADAPTIVE', True)
        use_cadpa = getattr(cadlgi_cfg, 'USE_CADPA', True)
        max_prototypes = getattr(cadlgi_cfg, 'MAX_PROTOTYPES', 32)
        min_active_ratio = getattr(cadlgi_cfg, 'MIN_ACTIVE_RATIO', 0.25)
        label_smooth = getattr(cadlgi_cfg, 'LABEL_SMOOTH', 0.0)
        head_drop = getattr(cadlgi_cfg, 'HEAD_DROP', 0.2)
        use_pretrained = getattr(cadlgi_cfg, 'USE_PRETRAINED', True)
        pretrained_path = getattr(cadlgi_cfg, 'PRETRAINED_PATH', None)

    # An empty path disables pretrained initialization.
    if not use_pretrained:
        pretrained_path = ""
        print("Pretrained initialization disabled.")

    if use_algf:
        if algf_version == 2:
            print(f"ALGF enabled.")
            print(f"Spatial gating: {'enabled' if use_spatial_adaptive else 'disabled'}")
        else:
            print(f"Static local-global fusion enabled.")

    model = CADLGIRep(
        backbone_type=backbone_type,
        input_size=cfg.DATA.CROPSIZE,
        num_classes=cfg.DATA.NUMBER_CLASSES,
        num_heads=num_heads,
        use_algf=use_algf,
        algf_version=algf_version,
        use_spatial_adaptive=use_spatial_adaptive,
        use_cadpa=use_cadpa,
        max_prototypes=max_prototypes,
        min_active_ratio=min_active_ratio,
        label_smooth=label_smooth,
        head_drop=head_drop,
        pretrained_path=pretrained_path,
    )

    return model

class PureRepViT(nn.Module):
    """RepViT classification baseline without the CADLGI block."""
    def __init__(
        self,
        backbone_type: str = 'repvit_m1_0',
        input_size: int = 224,
        num_classes: int = 6,
        label_smooth: float = 0.0,
        head_drop: float = 0.2,
        pretrained_path: Optional[str] = None,
    ):
        super().__init__()

        self.backbone_type = backbone_type
        self.input_size = input_size
        self.num_classes = num_classes

        REPVIT_MODELS = {
            'repvit_m0_9': repvit_m0_9,
            'repvit_m1_0': repvit_m1_0,
            'repvit_m1_1': repvit_m1_1,
            'repvit_m1_5': repvit_m1_5,
            'repvit_m2_3': repvit_m2_3,
        }

        assert backbone_type in REPVIT_MODELS, f"Unknown backbone: {backbone_type}"

        self.repvit = REPVIT_MODELS[backbone_type](
            pretrained=False,
            num_classes=num_classes,
            distillation=False
        )

        REPVIT_DIMS = {
            'repvit_m0_9': 384,
            'repvit_m1_0': 448,
            'repvit_m1_1': 448,
            'repvit_m1_5': 512,
            'repvit_m2_3': 640,
        }
        self.dim = REPVIT_DIMS.get(backbone_type, 448)

        self.head_drop_rate = head_drop
        self.head_dropout = nn.Dropout(head_drop) if head_drop > 0 else nn.Identity()

        self.label_smooth = label_smooth
        if label_smooth > 0:
            self.ce = LabelSmoothingCrossEntropy(smoothing=label_smooth)
        else:
            self.ce = nn.CrossEntropyLoss()

        self.pretrained_info = {'mode': 'scratch', 'backbone': backbone_type}

        if pretrained_path is None:
            pretrained_path = "auto"

        if pretrained_path:
            self._load_pretrained(pretrained_path)

        # For trainer compatibility
        self.side = None
        self.supports_label_loss = True  # Tell trainer to pass labels

    def _load_pretrained(self, path: str):
        self.pretrained_info = load_repvit_pretrained(
            self.repvit, path, self.backbone_type, REPVIT_PRETRAINED_URLS)

    def forward(self, x: Tensor, label: Optional[Tensor] = None):
        """Return logits and optional classification loss."""
        features = x
        for block in self.repvit.features:
            features = block(features)
        features = features.mean(dim=(2, 3))
        logits = self.repvit.classifier(self.head_dropout(features))

        if self.training and label is not None:
            loss = self.ce(logits, label)
            return logits, loss

        return logits

    def get_param_count(self) -> dict:
        """Return total and trainable parameter counts."""
        total = sum(p.numel() for p in self.parameters())
        backbone = sum(p.numel() for n, p in self.named_parameters() if 'head' not in n)
        head = sum(p.numel() for n, p in self.named_parameters() if 'head' in n)

        return {
            'total': total,
            'total_M': total / 1e6,
            'backbone': backbone,
            'backbone_M': backbone / 1e6,
            'head': head,
            'head_M': head / 1e6,
        }

def build_pure_repvit(cfg):
    """Build the RepViT classification baseline."""
    cadlgi_cfg = getattr(cfg.MODEL, 'CADLGI', None)

    if cadlgi_cfg is None:
        backbone_type = 'repvit_m1_0'
        label_smooth = 0.0
        head_drop = 0.2
        use_pretrained = True
        pretrained_path = None
    else:
        backbone_type = getattr(cadlgi_cfg, 'BACKBONE_TYPE', 'repvit_m1_0')
        label_smooth = getattr(cadlgi_cfg, 'LABEL_SMOOTH', 0.0)
        head_drop = getattr(cadlgi_cfg, 'HEAD_DROP', 0.2)
        use_pretrained = getattr(cadlgi_cfg, 'USE_PRETRAINED', True)
        pretrained_path = getattr(cadlgi_cfg, 'PRETRAINED_PATH', None)

    if not use_pretrained:
        pretrained_path = ""
        print("Pretrained initialization disabled.")

    model = PureRepViT(
        backbone_type=backbone_type,
        input_size=cfg.DATA.CROPSIZE,
        num_classes=cfg.DATA.NUMBER_CLASSES,
        label_smooth=label_smooth,
        head_drop=head_drop,
        pretrained_path=pretrained_path,
    )

    return model
