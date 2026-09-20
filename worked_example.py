"""Walk one training example through a tiny network and print every number.

The network is small enough (2 inputs, 2 hidden units, 2 classes) that you
can redo every step with a pencil. The same functions used for MNIST are
used here, so if these numbers are right, the big network is right too.

Run from this folder: python worked_example.py
"""

import numpy as np

from nn.activations import relu, relu_backward
from nn.layers import DenseParams, dense_backward, dense_forward
from nn.loss import cross_entropy, softmax, softmax_cross_entropy_backward

np.set_printoptions(precision=4, suppress=True)

# One example with two features, and the label is class 1.
X = np.array([[1.0, 2.0]])
y = np.array([[0.0, 1.0]])

layer1 = DenseParams(W=np.array([[0.1, -0.2], [0.3, 0.4]]), b=np.array([0.0, 0.0]))
layer2 = DenseParams(W=np.array([[0.5, -0.5], [0.1, 0.2]]), b=np.array([0.0, 0.0]))


def show(name: str, value) -> None:
    print(f"{name:<10}= {np.array2string(np.asarray(value), prefix=' ' * 12)}")


print("=== FORWARD PASS ===")
show("X", X)
Z1 = dense_forward(X, layer1)
show("Z1 = XW1+b1", Z1)
A1 = relu(Z1)
show("A1 = relu", A1)
Z2 = dense_forward(A1, layer2)
show("Z2 = A1W2+b2", Z2)
probs = softmax(Z2)
show("p = softmax", probs)
show("y", y)
print(f"loss      = {cross_entropy(probs, y):.4f}   (= -log p[correct class])")

print("\n=== BACKWARD PASS (chain rule, right to left) ===")
dZ2 = softmax_cross_entropy_backward(probs, y)
show("dL/dZ2", dZ2)
print("          ^ this is simply p - y")
dW2, db2, dA1 = dense_backward(dZ2, A1, layer2)
show("dL/dW2", dW2)
print("          ^ A1^T @ dZ2")
show("dL/db2", db2)
show("dL/dA1", dA1)
print("          ^ dZ2 @ W2^T")
dZ1 = relu_backward(dA1, Z1)
show("dL/dZ1", dZ1)
print("          ^ passes through where Z1 > 0, zero elsewhere")
dW1, db1, _ = dense_backward(dZ1, X, layer1)
show("dL/dW1", dW1)
print("          ^ X^T @ dZ1")
show("dL/db1", db1)

print("\n=== ONE GRADIENT DESCENT STEP (lr = 0.5) ===")
lr = 0.5
new_layer2 = DenseParams(W=layer2.W - lr * dW2, b=layer2.b - lr * db2)
new_layer1 = DenseParams(W=layer1.W - lr * dW1, b=layer1.b - lr * db1)
new_probs = softmax(dense_forward(relu(dense_forward(X, new_layer1)), new_layer2))
show("new p", new_probs)
print(f"new loss  = {cross_entropy(new_probs, y):.4f}   (was {cross_entropy(probs, y):.4f})")
print("Probability of the correct class went up. That is all training is.")
