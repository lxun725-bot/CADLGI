"""Require complete RepViT feature weights for a declared pretrained run."""
import hashlib
from pathlib import Path
import urllib.request
from urllib.parse import urlparse
import torch


def load_repvit_pretrained(model, path, backbone_type, urls):
    if str(path).lower() == 'auto':
        path = urls[backbone_type]
    if str(path).startswith(('https://', 'http://')):
        cache = Path.home() / '.cache' / 'repvit'
        cache.mkdir(parents=True, exist_ok=True)
        destination = cache / Path(urlparse(str(path)).path).name
        if not destination.is_file():
            temporary = destination.with_suffix(destination.suffix + '.download')
            # A failed request cannot leave a partial file masquerading as a cache hit.
            urllib.request.urlretrieve(str(path), temporary)
            temporary.replace(destination)
        path = destination
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f'Pretrained weights are required but missing: {path}')
    checkpoint = torch.load(path, map_location='cpu')
    source = checkpoint.get('model', checkpoint.get('state_dict', checkpoint))
    expected = model.state_dict()
    feature_keys = {key for key in expected if key.startswith('features.')}
    matched = {}
    for key, value in source.items():
        while key.startswith(('module.', 'backbone.', 'repvit.')):
            key = key.split('.', 1)[1]
        if key in feature_keys:
            if key in matched:
                raise ValueError(f'Duplicate pretrained feature key: {key}')
            if value.shape != expected[key].shape:
                raise ValueError(f'Pretrained feature shape mismatch: {key}')
            matched[key] = value
    # Old PyTorch files may omit counters; learned weights and BN statistics are mandatory.
    missing = feature_keys - matched.keys()
    required_missing = sorted(key for key in missing if not key.endswith('num_batches_tracked'))
    if required_missing:
        raise ValueError(f'Incomplete pretrained backbone ({len(required_missing)} missing tensors): '
                         f'{required_missing[:8]}')
    expected.update(matched)
    model.load_state_dict(expected, strict=True)
    info = {'mode': 'pretrained', 'backbone': backbone_type, 'path': str(path),
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'loaded_feature_tensors': len(matched), 'expected_feature_tensors': len(feature_keys),
            'defaulted_batch_counters': sorted(missing)}
    print(f'Loaded RepViT feature weights: {len(matched)}/{len(feature_keys)} tensors; SHA256={info["sha256"]}')
    return info
