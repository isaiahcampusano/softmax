"""A small, numerically stable softmax implementation."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def softmax(logits: ArrayLike, axis: int = -1) -> NDArray[np.float64]:
    """Convert logits into weights that sum to one along ``axis``.

    Args:
        logits: A non-empty, finite, NumPy-compatible array of raw scores.
        axis: The axis whose values should be normalized. The last axis is
            used by default.

    Returns:
        A float64 array with the same shape as ``logits``.

    Raises:
        ValueError: If ``logits`` is scalar, empty, or contains a non-finite
            value.
    """
    values = np.asarray(logits, dtype=np.float64)

    if values.ndim == 0:
        raise ValueError("logits must contain at least one dimension")
    if values.size == 0:
        raise ValueError("logits must not be empty")
    if not np.all(np.isfinite(values)):
        raise ValueError("logits must contain only finite values")

    shifted = values - np.max(values, axis=axis, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / np.sum(exponentials, axis=axis, keepdims=True)
