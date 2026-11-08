"""Combine directed scores into reciprocal ranking scores."""

import numpy as np


def penalize(product, likes_received, alpha):
    if not np.isfinite(alpha) or alpha < 0:
        raise ValueError("alpha must be finite and nonnegative")
    return product / (likes_received + 1.0) ** alpha


def fuse(forward, reverse, likes_received, alpha):
    total = forward + reverse
    product = forward * reverse
    harmonic = np.divide(2 * product, total, out=np.zeros_like(product), where=total > 0)
    return {
        "one_way": forward,
        "harmonic": harmonic,
        "product": product,
        "product_debiased": penalize(product, likes_received, alpha),
    }
