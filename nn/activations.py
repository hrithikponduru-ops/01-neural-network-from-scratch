"""Element-wise activation functions and their derivatives."""

import numpy as np


def relu(z: np.ndarray) -> np.ndarray:
    return np.maximum(z, 0.0)


def relu_backward(grad_out: np.ndarray, z: np.ndarray) -> np.ndarray:
    """d(loss)/dz given d(loss)/d(relu(z)).

    relu'(z) is 1 where z > 0 and 0 elsewhere, so the upstream gradient
    passes through wherever the unit was active and is blocked otherwise.
    """
    return grad_out * (z > 0.0)