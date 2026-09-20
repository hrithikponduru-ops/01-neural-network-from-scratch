"""Verify hand-derived gradients two independent ways.

1. Finite differences: perturb one parameter by +/- h and watch the loss.
   This is the definition of the derivative and depends on nothing but the
   forward pass, so it catches errors in the derivation itself.
2. PyTorch autograd: build the identical network in torch and compare.
   This catches errors that finite differences might miss due to rounding.

If the analytic gradient is wrong, training may still *appear* to work
(loss goes down a bit) while being subtly broken. These tests are the
only thing that proves the math is right.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nn.data import one_hot  # noqa: E402
from nn.network import (  # noqa: E402
    DenseParams,
    Network,
    backward,
    forward,
    init_network,
    loss,
)

IN, HIDDEN, CLASSES, BATCH = 20, 7, 10, 5

@pytest.fixture
def small_problem():
    rng = np.random.default_rng(0)
    net = init_network(IN, HIDDEN, CLASSES, rng)
    X = rng.normal(size=(BATCH, IN))
    y = one_hot(rng.integers(0, CLASSES, size=BATCH), CLASSES)
    return net, X, y

def _finite_difference(net: Network, X, y, pick, h: float = 1e-5) -> np.ndarray:
    """Central difference dL/dtheta for every entry of the array `pick(net)` returns."""
    theta = pick(net)
    grad = np.zeros_like(theta)
    for idx in np.ndindex(theta.shape):
        bump = np.zeros_like(theta)
        bump[idx] = h
        plus = _with_param(net, pick, theta + bump)
        minus = _with_param(net, pick, theta - bump)
        grad[idx] = (loss(plus, X, y) - loss(minus, X, y)) / (2 * h)
    return grad

def _with_param(net: Network, pick, new_value) -> Network:
    """Return a copy of net with the array selected by `pick` replaced."""
    if pick(net) is net.layer1.W:
        return Network(DenseParams(new_value, net.layer1.b), net.layer2)
    if pick(net) is net.layer1.b:
        return Network(DenseParams(net.layer1.W, new_value), net.layer2)
    if pick(net) is net.layer2.W:
        return Network(net.layer1, DenseParams(new_value, net.layer2.b))
    return Network(net.layer1, DenseParams(net.layer2.W, new_value))

@pytest.mark.parametrize(
    "pick",
    [
        lambda n: n.layer1.W,
        lambda n: n.layer1.b,
        lambda n: n.layer2.W,
        lambda n: n.layer2.b,
    ],
    ids=["W1", "b1", "W2", "b2"],
)
def test_analytic_matches_finite_difference(small_problem, pick):
    net, X, y = small_problem
    _, cache = forward(net, X)
    analytic = pick(backward(net, cache, y))
    numeric = _finite_difference(net, X, y, pick)
    np.testing.assert_allclose(analytic, numeric, rtol=1e-5, atol=1e-7)

def test_analytic_matches_torch_autograd(small_problem):
    net, X, y = small_problem
    _, cache = forward(net, X)
    grads = backward(net, cache, y)

    W1 = torch.tensor(net.layer1.W, requires_grad=True)
    b1 = torch.tensor(net.layer1.b, requires_grad=True)
    W2 = torch.tensor(net.layer2.W, requires_grad=True)
    b2 = torch.tensor(net.layer2.b, requires_grad=True)
    Xt = torch.tensor(X)
    yt = torch.tensor(y.argmax(axis=1))

    logits = torch.relu(Xt @ W1 + b1) @ W2 + b2
    torch.nn.functional.cross_entropy(logits, yt).backward()

    np.testing.assert_allclose(grads.layer1.W, W1.grad.numpy(), atol=1e-10)
    np.testing.assert_allclose(grads.layer1.b, b1.grad.numpy(), atol=1e-10)
    np.testing.assert_allclose(grads.layer2.W, W2.grad.numpy(), atol=1e-10)
    np.testing.assert_allclose(grads.layer2.b, b2.grad.numpy(), atol=1e-10)
