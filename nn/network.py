"""A two-layer network: Dense -> ReLU -> Dense -> softmax cross-entropy."""

from dataclasses import dataclass

import numpy as np

from .activations import relu, relu_backward
from .layers import DenseParams, dense_backward, dense_forward, init_dense
from .loss import cross_entropy, softmax, softmax_cross_entropy_backward


@dataclass(frozen=True)
class Network:
    layer1: DenseParams
    layer2: DenseParams


@dataclass(frozen=True)
class Cache:
    """Everything the backward pass needs from the forward pass."""
    X: np.ndarray
    Z1: np.ndarray
    A1: np.ndarray
    probs: np.ndarray


def init_network(
    in_features: int, hidden: int, num_classes: int, rng: np.random.Generator
) -> Network:
    return Network(
        layer1=init_dense(in_features, hidden, rng),
        layer2=init_dense(hidden, num_classes, rng),
    )


def forward(net: Network, X: np.ndarray) -> tuple[np.ndarray, Cache]:
    Z1 = dense_forward(X, net.layer1)
    A1 = relu(Z1)
    Z2 = dense_forward(A1, net.layer2)
    probs = softmax(Z2)
    return probs, Cache(X=X, Z1=Z1, A1=A1, probs=probs)


def loss(net: Network, X: np.ndarray, y_onehot: np.ndarray) -> float:
    probs, _ = forward(net, X)
    return cross_entropy(probs, y_onehot)


def backward(net: Network, cache: Cache, y_onehot: np.ndarray) -> Network:
    """Return a Network-shaped object holding gradients instead of parameters."""
    dZ2 = softmax_cross_entropy_backward(cache.probs, y_onehot)
    dW2, db2, dA1 = dense_backward(dZ2, cache.A1, net.layer2)
    dZ1 = relu_backward(dA1, cache.Z1)
    dW1, db1, _ = dense_backward(dZ1, cache.X, net.layer1)
    return Network(
        layer1=DenseParams(W=dW1, b=db1),
        layer2=DenseParams(W=dW2, b=db2),
    )


def sgd_step(net: Network, grads: Network, lr: float) -> Network:
    """Plain gradient descent. Returns a new Network; nothing is mutated."""
    return Network(
        layer1=DenseParams(
            W=net.layer1.W - lr * grads.layer1.W,
            b=net.layer1.b - lr * grads.layer1.b,
        ),
        layer2=DenseParams(
            W=net.layer2.W - lr * grads.layer2.W,
            b=net.layer2.b - lr * grads.layer2.b,
        ),
    )


def predict(net: Network, X: np.ndarray) -> np.ndarray:
    probs, _ = forward(net, X)
    return probs.argmax(axis=1)