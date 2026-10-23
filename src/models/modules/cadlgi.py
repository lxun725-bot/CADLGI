"""Parallel fusion and prototype cross-attention."""
import math
from typing import Optional, Tuple, List, Union
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from .algf import ALGF, StaticLocalGlobalFusion
from .cadpa import CADPA

class CADLGIBlock(nn.Module):
    """CADLGI cross-attention block with parallel CADPA and ALGF branches."""
    def __init__(
        self,
        dim: int,
        query_size: Tuple[int, int],
        num_heads: int = 4,
        use_algf: bool = True,
        algf_version: int = 2,
        use_spatial_adaptive: bool = True,
        use_cadpa: bool = True,
        max_prototypes: int = 32,
        min_active_ratio: float = 0.25,
    ):
        super().__init__()
        self.dim = dim
        self.query_size = query_size
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.use_algf = use_algf
        self.algf_version = algf_version
        self.use_cadpa = use_cadpa
        self.max_prototypes = max_prototypes

        # Enhanced query branch.
        if use_algf:
            if algf_version == 2:
                self.algf = ALGF(
                    dim=dim,
                    use_spatial_adaptive=use_spatial_adaptive
                )
                print(f"Query branch: ALGF.")
            else:
                self.algf = StaticLocalGlobalFusion(dim=dim)
                print(f"Query branch: static local-global fusion.")

        # Prototype memory branch.
        if use_cadpa:
            self.cadpa = CADPA(dim, query_size, max_prototypes, min_active_ratio)
            print(f"CADPA memory: K={max_prototypes}.")
        else:
            # Matched K learned candidates without CCE or gating. A single mean
            # prototype would make attention identically 1 and kill Q/ALGF gradients.
            self.static_prototype_generator = nn.Linear(dim, max_prototypes, bias=False)

        # Cross-attention projections.
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.norm_out = nn.LayerNorm(dim)

        # Dropout
        self.attn_drop = nn.Dropout(0.1)
        self.proj_drop = nn.Dropout(0.1)

        # Fixed residual scale.
        self.register_buffer('residual_scale', torch.ones(1))

        self._dynamic_stats = None
        self._attention_weights = None

    def forward(
        self,
        x: torch.Tensor,
        return_dynamic_stats: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, dict]]:
        """Combine enhanced queries and prototype memory through cross-attention."""
        b, n, c = x.shape
        residual = x

        # Compute the parallel query and memory branches.

        if self.use_algf:
            q_enhanced = self.algf(x, self.query_size)  # [B, N, C]
        else:
            q_enhanced = x

        if self.use_cadpa:
            p_dyn, gate_mask, stats = self.cadpa(x)
            self._dynamic_stats = stats
        else:
            weights = self.static_prototype_generator(x).transpose(1, 2).softmax(dim=-1)
            p_dyn = weights @ x
            gate_mask = x.new_ones(b, self.max_prototypes)
            stats = {'num_active': self.max_prototypes, 'activation_rate': 1.0,
                     'complexity': None, 'gate_mode': 'static_all_candidates'}
            self._dynamic_stats = stats

        # Attend to the prototype memory.

        q = self.norm_q(q_enhanced)  # [B, N, C]
        # Query projection is shared; selected prototypes are projected below.
        q = self.q_proj(q)  # [B, N, C]
        q = q.view(b, n, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, N, D]
        if not self.training and self.use_cadpa:
            z, attn, lengths = self._sparse_prototype_attention(q, p_dyn, gate_mask)
            stats['execution_mode'] = 'grouped_sparse_eval'
        else:
            z, attn = self._attend(q, p_dyn, gate_mask)
            lengths = [p_dyn.shape[1]] * b
            stats['execution_mode'] = 'dense_soft_train' if self.use_cadpa else 'dense_static'
        stats['kv_tokens_per_sample'] = lengths
        self._attention_weights = attn.detach()
        z = z.transpose(1, 2).contiguous().view(b, n, c)  # [B, N, C]

        z = self.out_proj(z)
        z = self.proj_drop(z)

        # Residual connection and output normalization.
        z_out = residual + self.residual_scale * z

        z_out = self.norm_out(z_out)

        if return_dynamic_stats:
            return z_out, stats

        return z_out

    def _attend(self, q, prototypes, gate_mask=None):
        b, k, _ = prototypes.shape
        kv = self.norm_kv(prototypes)
        key = self.k_proj(kv).view(b, k, self.num_heads, self.head_dim).transpose(1, 2)
        value = self.v_proj(kv).view(b, k, self.num_heads, self.head_dim).transpose(1, 2)
        # Keep small attention logits/softmax in FP32 when AMP uses FP16.
        # The matrix products still receive gradients; float64 gradcheck is preserved.
        with torch.autocast(device_type=q.device.type, enabled=False):
            q_scores = q.float() if q.dtype in (torch.float16, torch.bfloat16) else q
            k_scores = key.float() if key.dtype in (torch.float16, torch.bfloat16) else key
            logits = (q_scores @ k_scores.transpose(-2, -1)) * self.scale
        if gate_mask is not None:
            # Apply the activation prior after LayerNorm, which otherwise largely
            # cancels multiplicative soft-gate scaling.
            log_gate = gate_mask.float().clamp_min(1e-8).log()
            log_gate = log_gate.masked_fill(gate_mask == 0, float('-inf'))
            logits = logits + log_gate[:, None, None, :].to(logits.dtype)
        attention = self.attn_drop(F.softmax(logits, dim=-1))
        return attention.to(value.dtype) @ value, attention

    def _sparse_prototype_attention(self, q, prototypes, gate_mask):
        # Group samples by their actual M. Gather BEFORE LayerNorm/K/V projection;
        # no inactive prototype enters those projections or the attention kernel.
        b, k, c = prototypes.shape
        counts = gate_mask.sum(-1).long()
        if (counts < 1).any():
            raise ValueError('Sparse attention requires at least one active prototype')
        output = torch.empty_like(q)
        full_attention = q.new_zeros(b, self.num_heads, q.shape[2], k)
        for count in counts.unique(sorted=True).tolist():
            batch = (counts == count).nonzero(as_tuple=True)[0]
            slots = gate_mask[batch].bool().nonzero(as_tuple=False)[:, 1].view(len(batch), count)
            chosen = prototypes[batch].gather(1, slots.unsqueeze(-1).expand(-1, -1, c))
            group_output, attention = self._attend(q[batch], chosen)
            output[batch] = group_output
            # Expand only the detached diagnostic cache, not the compute tensor.
            locations = slots[:, None, None, :].expand_as(attention)
            full_attention[batch] = full_attention[batch].scatter(-1, locations, attention.detach())
        return output, full_attention, counts.tolist()

    def get_dynamic_stats(self) -> Optional[dict]:
        """Return cached prototype diagnostics."""
        return self._dynamic_stats

    def get_attention_weights(self) -> Optional[torch.Tensor]:
        """Return cached attention weights shaped [B, heads, N, K]."""
        return self._attention_weights

    def get_algf_weight_stats(self) -> Optional[dict]:
        """Return cached ALGF fusion statistics."""
        if not self.use_algf:
            return None

        if self.algf_version == 2 and hasattr(self.algf, 'get_weight_stats'):
            return self.algf.get_weight_stats()

        return {'info': 'Static fusion has no dynamic gate statistics.'}
