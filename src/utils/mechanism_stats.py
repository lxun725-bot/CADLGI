"""Per-image diagnostic values collected from the same inference pass."""
import numpy as np


def collect_mechanism_stats(model, batch_size):
    model = model.module if hasattr(model, 'module') else model
    result = {name: np.full(batch_size, np.nan) for name in
              ('active_prototypes', 'complexity', 'fusion_weight_mean')}
    block = getattr(model, 'cadlgi', None)
    if block is None:
        return result
    if block.use_cadpa:
        stats = block.cadpa.dpg.get_stats()
        for source, name in [('num_active', 'active_prototypes'), ('complexity', 'complexity')]:
            value = stats.get(source)
            if value is not None:
                result[name] = value.detach().float().cpu().numpy().reshape(batch_size)
    else:
        result['active_prototypes'].fill(block.max_prototypes)
    fusion = getattr(block, 'algf', None)
    weights = getattr(fusion, '_cached_final_weights', None)
    if weights is not None:
        result['fusion_weight_mean'] = weights.detach().float().mean((1, 2)).cpu().numpy()
    return result
