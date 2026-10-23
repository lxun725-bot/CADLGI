"""Backbones retained for manuscript classification experiments."""
from pathlib import Path
import hashlib
import numpy as np
import torch
from .vit_backbones.swin_transformer import SwinTransformer
from .vit_backbones.vit import VisionTransformer

def build_swin_model(model_type, crop_size, prompt_cfg, model_root,
                     adapter_cfg=None, lora_cfg=None, cross_layer=False):
    if model_type not in {"swint_imagenet", "swint_imagenet_224"}:
        raise ValueError(f"Only the paper Swin-T is retained, got {model_type}")
    model = SwinTransformer(img_size=crop_size, embed_dim=96, depths=[2, 2, 6, 2],
                            num_heads=[3, 6, 12, 24], window_size=7,
                            num_classes=-1, cross_layer=cross_layer)
    model.pretrained_info = {'mode': 'scratch', 'backbone': model_type}
    if model_root:
        path = Path(model_root) / "swin_tiny_patch4_window7_224.pth"
        if not path.is_file():
            raise FileNotFoundError(f"Swin pretrained weights missing: {path}. Set MODEL.MODEL_ROOT '' for scratch construction.")
        checkpoint = torch.load(path, map_location="cpu")
        state = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
        features = {k: v for k, v in state.items() if not k.startswith('head.')}
        model.load_state_dict(features, strict=True)
        model.pretrained_info = {'mode': 'pretrained', 'backbone': model_type,
                                 'path': str(path.resolve()),
                                 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                                 'loaded_backbone_tensors': len(features),
                                 'expected_backbone_tensors': len(model.state_dict())}
    return model, 768

def build_vit_sup_models(model_type, crop_size, prompt_cfg=None, model_root=None,
                         adapter_cfg=None, lora_cfg=None, load_pretrain=True, vis=False):
    if model_type not in {"sup_vitb16", "sup_vitb16_224", "sup_vitb16_imagenet21k"}:
        raise ValueError(f"Only the paper ViT-B16 is retained, got {model_type}")
    model = VisionTransformer("sup_vitb16", img_size=crop_size, num_classes=0, vis=vis)
    if load_pretrain and model_root:
        filename = "imagenet21k_ViT-B_16.npz" if "imagenet21k" in model_type else "ViT-B_16.npz"
        path = Path(model_root) / filename
        if not path.is_file():
            raise FileNotFoundError(f"ViT pretrained weights missing: {path}")
        with np.load(path) as weights:
            model.load_from(weights)
    return model, 768
