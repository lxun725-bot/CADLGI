"""Differentiable training budget and exact inference selection for CADPA."""
import torch
from torch import nn


class DPG(nn.Module):
    """Interpolate a threshold in continuous rank space; select exact M at eval."""

    def __init__(self, dim, max_prototypes=32, min_active_ratio=0.25, temperature=0.5):
        super().__init__()
        if max_prototypes < 2 or not 0 < min_active_ratio <= 1 or temperature <= 0:
            raise ValueError('Invalid prototype budget or temperature')
        self.dim = dim
        self.max_prototypes = max_prototypes
        self.min_active_ratio = min_active_ratio
        self.temperature = temperature
        self.importance_proj = nn.Linear(dim, 1, bias=False)
        # Paper Eq. 12: optimize the scalar temperature factor.
        self.learnable_temp = nn.Parameter(torch.tensor(1.0))
        self._cached_mask = self._cached_num_active = self._cached_scores = None
        self._cached_budget = self._cached_complexity = self._cached_target = None

    def forward(self, prototypes, complexity, training=True):
        b, k, c = prototypes.shape
        importance = self.importance_proj(prototypes).squeeze(-1)
        min_active = max(1, int(k * self.min_active_ratio))
        budget = (complexity * k).clamp(min=min_active, max=k)
        target = budget.floor().long()
        scores, indices = importance.sort(dim=-1, descending=True, stable=True)
        if training:
            # Crucially, budget remains FLOAT here. Indexing picks neighboring
            # values, while the interpolation fraction carries gradients to CCE.
            position = (budget - 1).clamp(0, k - 1)
            lower = position.floor().long()
            upper = (lower + 1).clamp(max=k - 1)
            fraction = position - lower.to(position.dtype)
            threshold = scores.gather(1, lower) * (1 - fraction) + scores.gather(1, upper) * fraction
            effective_temp = self.temperature * torch.sigmoid(self.learnable_temp)
            soft_mask = torch.sigmoid((importance - threshold) / effective_temp)
            boost = torch.zeros_like(soft_mask).scatter(1, indices[:, :min_active], 0.5)
            mask = (soft_mask + boost).clamp(max=1)
        else:
            # Select rank positions instead of >= threshold; ties cannot inflate M.
            selected_ranks = torch.arange(k, device=prototypes.device)[None, :] < target
            mask = torch.zeros_like(importance).scatter(1, indices, selected_ranks.to(importance.dtype))
        self._cached_mask = mask.detach()
        self._cached_num_active = mask.sum(-1).detach()
        self._cached_scores = importance.detach()
        self._cached_budget = budget.detach()
        self._cached_complexity = complexity.detach()
        self._cached_target = target.detach()
        stats = {'num_active': self._cached_num_active.mean().item(),
                 'activation_rate': (self._cached_num_active.mean() / k).item(),
                 'target_active': target.float().mean().item(),
                 'complexity': complexity.detach().mean().item(),
                 'continuous_budget': budget.detach().mean().item(),
                 'soft_mass': self._cached_num_active.mean().item() if training else None,
                 'hard_M': self._cached_num_active.mean().item() if not training else None,
                 'gate_mode': 'soft_interpolated' if training else 'hard_exact_top_m'}
        return prototypes * mask.unsqueeze(-1), mask, stats

    def get_stats(self):
        return {'gate_mask': self._cached_mask, 'num_active': self._cached_num_active,
                'importance_scores': self._cached_scores, 'continuous_budget': self._cached_budget,
                'complexity': self._cached_complexity, 'target_M': self._cached_target}
