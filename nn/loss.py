"""Softmax cross-entropy loss and its gradient."""

import numpy as np


def softmax(logits: np.ndarray) -> np.ndarray:
    """Row-wise softmax with the max subtracted for numerical stability.

    exp(z) overflows float64 for z > ~709. Subtracting the row max leaves
    the probabilities unchanged (the constant cancels in the ratio) but
    keeps every exponent <= 0.
    """
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def cross_entropy(probs: np.ndarray, y_onehot: np.ndarray) -> float:
    """Mean negative log-likelihood of the correct class over the batch."""
    n = probs.shape[0]
    eps = 1e-12
    return float(-np.sum(y_onehot * np.log(probs + eps)) / n)


def softmax_cross_entropy_backward(probs: np.ndarray, y_onehot: np.ndarray) -> np.ndarray:
    """d(mean loss)/d(logits).

    Combining softmax and cross-entropy gives the famously clean result
    (p - y) / n. Derived in README.md, section "Loss gradient".
    """
    n = probs.shape[0]
    return (probs - y_onehot) / n