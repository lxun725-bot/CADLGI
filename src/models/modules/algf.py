"""Local and global fusion branches."""
import math
from typing import Optional, Tuple, List, Union
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

class SpatialAveragePool(nn.Module):
    """Pool the complete map to ceil(H/r) x ceil(W/r) spatial positions."""
    def __init__(self, ratio):
        super().__init__()
        if ratio < 1:
            raise ValueError('Pooling ratio must be positive')
        self.ratio = ratio

    def forward(self, x):
        h, w = x.shape[-2:]
        return F.adaptive_avg_pool2d(x, (math.ceil(h / self.ratio), math.ceil(w / self.ratio)))

class StaticLocalGlobalFusion(nn.Module):
    """Static local-global fusion control."""

    def __init__(self, dim: int, local_kernel: int = 3, downsample_ratio: int = 4):
        """Initialize the local and global fusion branches."""
        super().__init__()
        self.local_kernel = local_kernel
        self.downsample_ratio = downsample_ratio

        # Local feature extraction.
        self.local_conv = nn.Sequential(
            nn.Conv2d(dim, dim, local_kernel, padding=local_kernel//2, groups=dim),
            nn.GroupNorm(min(32, dim), dim),
            nn.GELU(),
            # Mix channels after depthwise convolution.
            nn.Conv2d(dim, dim, 1),
            nn.GroupNorm(min(32, dim), dim)
        )


        # Global feature extraction at reduced resolution.
        self.global_downsample = nn.AvgPool2d(downsample_ratio)

        num_heads = min(8, dim // 64)
        self.global_attention = nn.MultiheadAttention(
            dim,
            num_heads=num_heads,
            batch_first=True
        )

        self.global_upsample = nn.Upsample(
            scale_factor=downsample_ratio,
            mode='bilinear',
            align_corners=False
        )

        # Map concatenated features to the input width.
        self.fusion = nn.Sequential(
            nn.Linear(dim * 2, dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim * 2, dim)
        )

        # Shared residual fusion coefficient.
        self.fusion_weight = nn.Parameter(torch.tensor(0.5))

    def forward(self, x: torch.Tensor, hw: Tuple[int, int]) -> torch.Tensor:
        """Fuse tokens shaped [B, H*W, C] with a residual connection."""
        b, n, c = x.shape
        h, w = hw

        if h * w != n:
            return x

        # transpose(1, 2): [B, N, C] → [B, C, N]
        # view(b, c, h, w): [B, C, N] → [B, C, H, W]
        x_2d = x.transpose(1, 2).view(b, c, h, w)

        local_feat = self.local_conv(x_2d)

        # Extract local features.
        local_feat = local_feat.view(b, c, n).transpose(1, 2)  # [B, N, C]


        global_feat = self.global_downsample(x_2d)  # [B, C, gh, gw]
        gh, gw = global_feat.shape[-2:]

        # [B, C, gh, gw] → [B, gh*gw, C]
        global_feat_flat = global_feat.view(b, c, gh * gw).transpose(1, 2)  # [B, gh*gw, C]

        # Extract global features.
        global_feat_attn, _ = self.global_attention(
            global_feat_flat,  # Query
            global_feat_flat,  # Key
            global_feat_flat  # Value
        )

        global_feat_2d = global_feat_attn.transpose(1, 2).view(b, c, gh, gw)

        global_feat_up = self.global_upsample(global_feat_2d)

        if global_feat_up.shape[-2:] != (h, w):
            global_feat_up = F.interpolate(
                global_feat_up,
                size=(h, w),
                mode="bilinear",
                align_corners=False
            )

        global_feat = global_feat_up.view(b, c, n).transpose(1, 2)  # [B, N, C]


        # [B, N, C] + [B, N, C] → [B, N, 2C]
        combined = torch.cat([local_feat, global_feat], dim=-1)  # [B, N, 2C]

        # Fuse the two branches.
        # [B, N, 2C] → [B, N, 2C] → [B, N, C]
        fused = self.fusion(combined)  # [B, N, C]

        output = x + self.fusion_weight * fused

        return output

class ALGF(nn.Module):
    """Adaptive Local–Global Fusion with complete-map, multi-token pooling."""

    def __init__(
        self,
        dim: int,
        local_kernel: int = 3,
        downsample_ratio: int = 4,
        use_spatial_adaptive: bool = True,
        weight_init: float = 0.5
    ):
        """Initialize the local and global fusion branches."""
        super().__init__()
        self.gate_mode = "joint" if use_spatial_adaptive else "sample"
        self.dim = dim
        self.local_kernel = local_kernel
        self.downsample_ratio = downsample_ratio
        self.use_spatial_adaptive = use_spatial_adaptive
        self.weight_init = weight_init

        # Local feature extraction.
        self.local_conv = nn.Sequential(
            nn.Conv2d(dim, dim, local_kernel, padding=local_kernel//2, groups=dim),
            nn.GroupNorm(min(32, dim), dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, 1),
            nn.GroupNorm(min(32, dim), dim)
        )

        # At Stage-4, 7x7 -> 2x2; adaptive bins include every edge position.
        self.global_downsample = SpatialAveragePool(downsample_ratio)
        num_heads = max(1, min(8, dim // 64))
        self.global_attention = nn.MultiheadAttention(
            dim,
            num_heads=num_heads,
            batch_first=True
        )
        self.global_upsample = nn.Upsample(
            scale_factor=downsample_ratio,
            mode='bilinear',
            align_corners=False
        )

        self.fusion = nn.Sequential(
            nn.Linear(dim * 2, dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim * 2, dim)
        )

        # Sample-level gate.
        hidden_dim = max(32, dim // 8)
        self.sample_weight_net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

        # Optional spatial gate.
        if use_spatial_adaptive:
            self.spatial_weight_net = nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid()
            )
        else:
            self.spatial_weight_net = None

        # Paper Eq. 17: trainable calibration, bounded output in forward.
        self.base_weight = nn.Parameter(torch.tensor(weight_init))
        self.dynamic_weight_scale = nn.Parameter(torch.tensor(0.3))

        self._cached_sample_weights = None
        self._cached_spatial_weights = None
        self._cached_final_weights = None
        self._cached_local_feat = None
        self._cached_global_feat = None

    def configure_gate_mode(self, mode):
        """Apply a predeclared mechanism control before optimizer construction."""
        if mode not in {'joint', 'sample', 'spatial', 'static', 'concat'}:
            raise ValueError(f'Unknown ALGF gate mode: {mode}')
        self.gate_mode = mode
        if mode not in {'joint', 'sample'}:
            self.sample_weight_net = None
        if mode not in {'joint', 'spatial'}:
            self.spatial_weight_net = None
        if mode in {'joint', 'spatial'} and self.spatial_weight_net is None:
            raise ValueError('Spatial gating requires USE_SPATIAL_ADAPTIVE=True')
        if mode in {'static', 'concat'}:
            value = self.dynamic_weight_scale.detach().clone()
            del self.dynamic_weight_scale
            self.register_buffer('dynamic_weight_scale', value)
        if mode == 'concat':
            value = self.base_weight.detach().clone()
            del self.base_weight
            self.register_buffer('base_weight', value)

    def forward(self, x: torch.Tensor, hw: Tuple[int, int]) -> torch.Tensor:
        """Fuse tokens shaped [B, H*W, C] with a residual connection."""
        b, n, c = x.shape
        h, w = hw

        if h * w != n:
            raise ValueError(f'ALGF spatial shape {hw} does not match {n} tokens')

        x_2d = x.transpose(1, 2).view(b, c, h, w)

        local_feat = self.local_conv(x_2d)
        local_feat = local_feat.view(b, c, n).transpose(1, 2)  # [B, N, C]

        global_feat = self.global_downsample(x_2d)
        gh, gw = global_feat.shape[-2:]

        global_feat_flat = global_feat.view(b, c, gh * gw).transpose(1, 2)
        global_feat_attn, _ = self.global_attention(
            global_feat_flat, global_feat_flat, global_feat_flat
        )

        global_feat_2d = global_feat_attn.transpose(1, 2).view(b, c, gh, gw)
        global_feat_up = self.global_upsample(global_feat_2d)

        if global_feat_up.shape[-2:] != (h, w):
            global_feat_up = F.interpolate(
                global_feat_up, size=(h, w), mode="bilinear", align_corners=False
            )

        global_feat = global_feat_up.view(b, c, n).transpose(1, 2)  # [B, N, C]

        combined = torch.cat([local_feat, global_feat], dim=-1)  # [B, N, 2C]
        fused = self.fusion(combined)  # [B, N, C]

        # Spatial response from local-global differences.

        global_context = x.mean(dim=1)
        sample_weights = (self.sample_weight_net(global_context) if self.sample_weight_net is not None
                          else x.new_ones(b, 1))

        if self.use_spatial_adaptive and self.spatial_weight_net is not None:
            feat_diff = torch.abs(local_feat - global_feat)  # [B, N, C]
            spatial_weights = self.spatial_weight_net(feat_diff)  # [B, N, 1]
        else:
            spatial_weights = torch.ones(b, n, 1, device=x.device)  # [B, N, 1]

        # final_weight = base_weight + scale * (sample_weight * spatial_weight - 0.5)
        dynamic_weight = sample_weights.unsqueeze(1) * spatial_weights  # [B, N, 1]
        final_weights = self.base_weight + self.dynamic_weight_scale * (dynamic_weight - 0.5)

        if self.gate_mode == 'static':
            final_weights = self.base_weight.expand(b, n, 1)
        final_weights = torch.clamp(final_weights, 0.1, 0.9)
        if self.gate_mode == 'concat':
            final_weights = x.new_ones(b, n, 1)

        # Inject the fused representation.
        # output = x + α_dynamic * fused
        output = x + final_weights * fused

        self._cached_sample_weights = sample_weights.detach()
        self._cached_spatial_weights = spatial_weights.detach()
        self._cached_final_weights = final_weights.detach()
        self._cached_local_feat = local_feat.detach()
        self._cached_global_feat = global_feat.detach()

        return output

    def get_weight_stats(self) -> dict:
        """Return cached fusion weights and scalar statistics."""
        if self._cached_final_weights is None:
            return {'error': 'No cached data. Run forward pass first.'}

        final_weights = self._cached_final_weights

        return {
            'sample_weights': self._cached_sample_weights,
            'spatial_weights': self._cached_spatial_weights,
            'final_weights': final_weights,
            'weight_mean': final_weights.mean().item(),
            'weight_std': final_weights.std().item(),
            'weight_min': final_weights.min().item(),
            'weight_max': final_weights.max().item(),
            'base_weight': self.base_weight.item(),
            'dynamic_scale': self.dynamic_weight_scale.item(),
        }

    def visualize_weights(self, hw: Tuple[int, int], batch_idx: int = 0) -> torch.Tensor:
        """Return the cached fusion weights as an H-by-W map."""
        if self._cached_final_weights is None:
            return None

        h, w = hw
        weights = self._cached_final_weights[batch_idx]  # [N, 1]
        weight_map = weights.view(h, w)  # [H, W]

        return weight_map
