"""Train the NumPy network on MNIST and save a loss/accuracy plot.

Run from this folder: python train.py
"""

import os

# Must come before NumPy is imported. With 64-row mini-batches the matrices
# are tiny, and a multithreaded BLAS spends far longer coordinating threads
# than multiplying. Measured on this machine: 0.32 s/batch with default
# threads vs 0.003 s/batch with one thread, a 100x difference.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from nn.data import load_mnist, one_hot
from nn.network import backward, forward, init_network, loss, predict, sgd_step

HIDDEN = 128
EPOCHS = 50
BATCH = 64
LR = 0.05
SEED = 0


def iterate_minibatches(X, y, batch, rng):
    order = rng.permutation(len(X))
    for start in range(0, len(X), batch):
        idx = order[start : start + batch]
        yield X[idx], y[idx]


def main() -> None:
    rng = np.random.default_rng(SEED)
    X_train, y_train = load_mnist(train=True)
    X_test, y_test = load_mnist(train=False)
    Y_train = one_hot(y_train)
    Y_test = one_hot(y_test)

    net = init_network(X_train.shape[1], HIDDEN, 10, rng)
    history = {"train_loss": [], "test_loss": [], "test_acc": []}

    for epoch in range(1, EPOCHS + 1):
        batch_losses = []
        for Xb, Yb in iterate_minibatches(X_train, Y_train, BATCH, rng):
            probs, cache = forward(net, Xb)
            grads = backward(net, cache, Yb)
            net = sgd_step(net, grads, LR)
            batch_losses.append(loss(net, Xb, Yb))

        test_loss = loss(net, X_test, Y_test)
        test_acc = float((predict(net, X_test) == y_test).mean())
        history["train_loss"].append(float(np.mean(batch_losses)))
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)
        print(
            f"epoch {epoch} train loss {history['train_loss'][-1]:.4f}  "
            f"test loss {test_loss:.4f}  test acc {test_acc:.4f}"
        )

    save_plot(history)


def save_plot(history: dict) -> None:
    epochs = range(1, len(history["test_acc"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(epochs, history["train_loss"], label="train")
    ax1.plot(epochs, history["test_loss"], label="test")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("cross-entropy loss")
    ax1.legend()
    ax2.plot(epochs, history["test_acc"])
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("test accuracy")
    fig.tight_layout()
    fig.savefig("figures/training.png", dpi=120)
    print("saved figures/training.png")


if __name__ == "__main__":
    main()
