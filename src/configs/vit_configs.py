"""ViT-B16 configuration retained for the paper baseline.
Copyright (c) Meta Platforms, Inc. All Rights Reserved.
"""
from .config_node import CfgNode

def get_b16_config():
    """Returns the ViT-B/16 configuration."""
    config = CfgNode()
    config.patches = CfgNode({'size': (16, 16)})
    config.hidden_size = 768
    config.transformer = CfgNode()
    config.transformer.mlp_dim = 3072
    config.transformer.num_heads = 12
    config.transformer.num_layers = 12
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1
    config.classifier = 'token'
    config.representation_size = None
    return config
