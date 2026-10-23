"""Checkpoint loading and parameter-name compatibility."""
from collections import OrderedDict
import hashlib
from pathlib import Path
import torch

IMPLEMENTATION_VERSION = 'cadlgi-paper-v3'

def get_source_fingerprint():
    root = Path(__file__).resolve().parents[2]
    digest = hashlib.sha256()
    paths = sorted((root / 'src').rglob('*.py')) + [root / 'run.py', root / 'test_best.py']
    for path in paths:
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes().replace(b'\r\n', b'\n'))
    return digest.hexdigest()

def get_model_contract(cfg):
    return {'type': cfg.MODEL.TYPE, 'feature': cfg.DATA.FEATURE, 'crop_size': cfg.DATA.CROPSIZE,
            'mlp_num': cfg.MODEL.MLP_NUM,
            'cadlgi': {k: v for k, v in cfg.MODEL.CADLGI.items()
                       if k not in {'USE_PRETRAINED', 'PRETRAINED_PATH'}}}


def migrate_legacy_state_dict(state_dict):
    """Rename keys only; parameter tensors and numerical values are untouched."""
    migrated = OrderedDict()
    for key, value in state_dict.items():
        if key.startswith('module.'):
            key = key[len('module.'):]
        parts = key.split('.')
        parts = ['cadlgi' if p == 'malt_attn' else 'algf' if p == 'local_global' else p
                 for p in parts]
        # Legacy CADPA children lived directly in the attention block.
        for old, new in [('complexity_estimator', ['cadpa', 'cce']),
                         ('prototype_gate', ['cadpa', 'dpg']),
                         ('prototype_generator', ['cadpa', 'prototype_generator'])]:
            if old in parts and 'cadpa' not in parts:
                i = parts.index(old)
                parts[i:i + 1] = new
        key = '.'.join(parts)
        if key in migrated:
            raise ValueError(f'Checkpoint key collision during migration: {key}')
        migrated[key] = value
    return migrated


def load_model_checkpoint(model, path, require_current=False):
    """Load complete model weights strictly, with legacy naming migration."""
    checkpoint = torch.load(path, map_location='cpu')
    if require_current and checkpoint.get('implementation_version') != IMPLEMENTATION_VERSION:
        raise ValueError('Checkpoint implementation identifier does not match the required identifier. '
                         'Use a checkpoint compatible with the selected loading mode.')
    state = checkpoint.get('model', checkpoint.get('state_dict', checkpoint))
    target = model.module if hasattr(model, 'module') else model
    target.load_state_dict(migrate_legacy_state_dict(state), strict=True)
    return checkpoint
