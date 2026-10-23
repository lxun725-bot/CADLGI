"""Transparent operator counting; never infer latency from prototype budgets."""
from fvcore.nn import FlopCountAnalysis


def count_model_operations(model, sample):
    analysis = FlopCountAnalysis(model, (sample,))
    analysis.unsupported_ops_warnings(False).uncalled_modules_warnings(False).tracer_warnings('none')
    total = analysis.total()
    return {
        'fvcore_counted_ops': int(total),
        'fvcore_counted_ops_G': total / 1e9,
        'fvcore_counting_convention': 'Supported operators only; one multiply-add counts as one operation',
        'fvcore_by_operator': dict(analysis.by_operator()),
        'fvcore_unsupported_ops': dict(analysis.unsupported_ops()),
    }
