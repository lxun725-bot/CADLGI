"""Content-aware prototype branch."""
import math
from typing import Optional, Tuple, List, Union
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from .prototype_gate import DPG

class CCE(nn.Module):
    """Content Complexity Estimator, trained through DPG's continuous threshold."""
    def __init__(self, dim: int, spatial_size: Tuple[int, int] = (7, 7)):
        super().__init__()
        self.dim = dim
        self.spatial_size = spatial_size

        # Spatial feature encoder.
        hidden_dim = min(dim // 8, 32)
        self.spatial_encoder = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False),
            nn.GroupNorm(min(32, dim), dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, hidden_dim, 1),
        )

        # Sample-level budget estimator.
        self.complexity_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

        # Learnable score offset.
        self.bias = nn.Parameter(torch.tensor(0.3))

    def forward(self, x: torch.Tensor, hw: Tuple[int, int]) -> torch.Tensor:
        """Estimate one budget-control score per input from spatial features."""
        b, n, c = x.shape
        h, w = hw

        if h * w != n:
            raise ValueError(f'CCE spatial shape {hw} does not match {n} tokens')
        x_2d = x.transpose(1, 2).view(b, c, h, w)

        spatial_feat = self.spatial_encoder(x_2d)

        raw_complexity = self.complexity_head(spatial_feat)  # [B, 1]

        # Calibrate the score to the bounded budget range.
        complexity = 0.15 + 0.7 * (raw_complexity * 0.6 + torch.sigmoid(self.bias) * 0.4)

        return complexity

class CADPA(nn.Module):
    """Content-Aware Dynamic Prototype Activation: CCE, generation, and DPG."""

    def __init__(self, dim, query_size, max_prototypes=32, min_active_ratio=0.25):
        super().__init__()
        self.query_size = query_size
        # Candidate aggregation is matched to the ungated multi-prototype control.
        self.cce = CCE(dim, query_size)
        self.prototype_generator = nn.Linear(dim, max_prototypes, bias=False)
        self.dpg = DPG(dim=dim, max_prototypes=max_prototypes,
                       min_active_ratio=min_active_ratio)

    def forward(self, x):
        complexity = self.cce(x, self.query_size)
        prototype_logits = self.prototype_generator(x)
        prototype_weights = F.softmax(prototype_logits.transpose(1, 2), dim=-1)
        prototypes = prototype_weights @ x
        p_dyn, gate_mask, stats = self.dpg(prototypes, complexity, training=self.training)
        return p_dyn, gate_mask, stats
