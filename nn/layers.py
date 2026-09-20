"""A fully connected (dense) layer with explicit forward and backward passes."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DenseParams:
    """Weights and biases. Frozen so updates return a new object."""
    W: np.ndarray  # shape (in_features, out_features)
    b: np.ndarray  # shape (out_features,)


def init_dense(in_features: int, out_features: int, rng: np.random.Generator) -> DenseParams:
    """He initialisation: variance 2/in_features keeps ReLU activations O(1)."""
    scale = np.sqrt(2.0 / in_features)
    W = rng.normal(0.0, scale, size=(in_features, out_features))
    b = np.zeros(out_features)
    return DenseParams(W=W, b=b)


def dense_forward(X: np.ndarray, params: DenseParams) -> np.ndarray:
    """Z = X W + b, with X of shape (n, in_features)."""
    return X @ params.W + params.b


def dense_backward(
    grad_out: np.ndarray, X: np.ndarray, params: DenseParams
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Backward pass for Z = X W + b.

    Args:
        grad_out: dL/dZ, shape (n, out_features). The gradient flowing back
            from whatever consumed this layer's output.
        X: the input that was fed to dense_forward, shape (n, in_features).
        params: the W and b used in the forward pass.

    Returns:
        (dL/dW, dL/db, dL/dX) with shapes matching W, b, and X respectively.

    Derivation (see README.md, "Dense layer gradient"). With G = dL/dZ and
    Z_ij = sum_k X_ik W_kj + b_j, the chain rule gives

        dL/dW_kj = sum_i G_ij X_ik        ->  dW = X^T G    (in, out)
        dL/db_j  = sum_i G_ij             ->  db = sum_i G  (out,)
        dL/dX_ik = sum_j G_ij W_kj        ->  dX = G W^T    (n, in)

    The sums over i appear because every row of the batch shares W and b.
    """
    dW = X.T @ grad_out
    db = grad_out.sum(axis=0)
    dX = grad_out @ params.W.T
    return dW, db, dX