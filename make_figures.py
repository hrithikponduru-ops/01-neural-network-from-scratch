"""Generate the explanatory figures used in README.md into figures/.

Run from this folder: python make_figures.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).parent / "figures"
OUT.mkdir(exist_ok=True)

INK = "#1f2937"
BLUE = "#2563eb"
GREEN = "#059669"
RED = "#dc2626"
AMBER = "#d97706"
GREY = "#9ca3af"

def box(ax, x, y, w, h, text, color, fontsize=10):
    ax.add_patch(
        FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                       linewidth=1.5, edgecolor=color, facecolor="white")
    )
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, color=INK, linespacing=1.5)

def arrow(ax, x0, y0, x1, y1, color, text=None, above=True, lw=1.8):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                 mutation_scale=14, linewidth=lw, color=color))
    if text:
        dy = 0.18 if above else -0.18
        ax.text((x0 + x1) / 2, (y0 + y1) / 2 + dy, text, ha="center",
                va="center", fontsize=9, color=color)


def fig_architecture():
    fig, ax = plt.subplots(figsize=(11, 3.6))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 3.6)
    ax.axis("off")
    stages = [
        ("Input\n784 pixels", GREY),
        ("Dense\nW1: 784x128\nb1: 128", BLUE),
        ("ReLU\nmax(z, 0)", AMBER),
        ("Dense\nW2: 128x10\nb2: 10", BLUE),
        ("Softmax\nprobabilities", GREEN),
        ("Cross-entropy\nloss", RED),
    ]
    w, h, gap = 1.45, 1.3, 0.35
    x = 0.2
    for i, (label, color) in enumerate(stages):
        box(ax, x, 1.4, w, h, label, color, fontsize=9)
        if i < len(stages) - 1:
            arrow(ax, x + w, 2.05, x + w + gap, 2.05, INK)
        x += w + gap
    names = ["X", "Z1", "A1", "Z2", "p", "L"]
    x = 0.2
    for name in names:
        ax.text(x + w / 2, 3.05, name, ha="center", fontsize=12,
                color=INK, style="italic", weight="bold")
        x += w + gap
    ax.text(5.5, 0.55, "Forward pass: data flows left to right, each box is a function",
            ha="center", fontsize=10, color=INK)
    ax.text(5.5, 0.15, "Backward pass: gradients flow right to left, each box multiplies by its derivative",
            ha="center", fontsize=10, color=RED)
    fig.savefig(OUT / "architecture.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_chain_rule():
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 4.2)
    ax.axis("off")
    ax.text(5.5, 3.95, "The chain rule as a bucket brigade", ha="center",
            fontsize=13, weight="bold", color=INK)
    fwd = [("Z1 = X·W1 + b1", BLUE), ("A1 = relu(Z1)", AMBER),
           ("Z2 = A1·W2 + b2", BLUE), ("L = CE(softmax(Z2), y)", RED)]
    bwd = ["dL/dZ1 = dL/dA1 ⊙ 1[Z1>0]", "dL/dA1 = dL/dZ2 · W2^T",
           "dL/dZ2 = (p - y)/n", "start: dL/dL = 1"]
    params = ["dL/dW1 = X^T · dL/dZ1\ndL/db1 = Σrows dL/dZ1", "",
              "dL/dW2 = A1^T · dL/dZ2\ndL/db2 = Σrows dL/dZ2", ""]
    w, h, gap = 2.3, 0.75, 0.45
    x = 0.2
    for i, (label, color) in enumerate(fwd):
        box(ax, x, 2.6, w, h, label, color, fontsize=9)
        if i < len(fwd) - 1:
            arrow(ax, x + w, 2.98, x + w + gap, 2.98, INK)
        x += w + gap
    x = 0.2
    for i, label in enumerate(bwd):
        box(ax, x, 1.45, w, h, label, RED, fontsize=8.5)
        if i < len(bwd) - 1:
            arrow(ax, x + w + gap, 1.82, x + w, 1.82, RED)
        x += w + gap
    x = 0.2
    for label in params:
        if label:
            box(ax, x, 0.2, w, 0.9, label, GREEN, fontsize=8.5)
            arrow(ax, x + w / 2, 1.45, x + w / 2, 1.1, GREEN)
        x += w + gap
    ax.text(0.2, 3.45, "FORWARD (compute the loss)", ha="left", fontsize=9,
            color=INK, style="italic")
    ax.text(0.2, 2.3, "BACKWARD (compute how the loss changes)", ha="left",
            fontsize=9, color=RED, style="italic")
    ax.text(5.5, 0.02, "PARAMETER GRADIENTS (what gradient descent uses)", ha="center",
            fontsize=9, color=GREEN, style="italic")
    fig.savefig(OUT / "chain_rule.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_gradient_descent():
    """A 1-D loss curve with descent steps, and a 2-D contour view."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    
    f = lambda w: (w - 2) ** 2 + 0.5
    df = lambda w: 2 * (w - 2)
    ws = np.linspace(-1.5, 5.5, 200)
    ax1.plot(ws, f(ws), color=INK, lw=2)
    w = 5.0
    for _ in range(7):
        ax1.plot(w, f(w), "o", color=RED, ms=7)
        w_next = w - 0.3 * df(w)
        ax1.annotate("", xy=(w_next, f(w_next)), xytext=(w, f(w)),
                     arrowprops=dict(arrowstyle="->", color=RED, lw=1.5))
        w = w_next
    ax1.set_xlabel("parameter w")
    ax1.set_ylabel("loss L(w)")
    ax1.set_title("w ← w - lr · dL/dw, repeated")
    ax1.annotate("minimum", xy=(2.0, 0.5), xytext=(0.2, 2.5), color=GREEN, fontsize=9,
                 arrowprops=dict(arrowstyle="->", color=GREEN, lw=1))
    ax1.spines[["top", "right"]].set_visible(False)

    a, b = np.meshgrid(np.linspace(-3, 3, 200), np.linspace(-3, 3, 200))
    L = a ** 2 + 4 * b ** 2
    ax2.contour(a, b, L, levels=14, colors=GREY, linewidths=0.8)
    p = np.array([2.6, 2.2])
    path = [p]
    for _ in range(12):
        grad = np.array([2 * p[0], 8 * p[1]])
        p = p - 0.1 * grad
        path.append(p)
    path = np.array(path)
    ax2.plot(path[:, 0], path[:, 1], "-o", color=RED, ms=4, lw=1.5)
    ax2.plot(0, 0, "*", color=GREEN, ms=14)
    ax2.set_xlabel("w₁")
    ax2.set_ylabel("w₂")
    ax2.set_title("Same idea with two parameters: step downhill")
    ax2.set_aspect("equal")
    ax2.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "gradient_descent.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_worked_example():
    """Draw the 2-2-2 toy network with the numbers from worked_example.py."""
    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 4.6)
    ax.axis("off")
    ax.text(5.5, 4.35, "Worked example: one input, two hidden units, two classes",
            ha="center", fontsize=13, weight="bold", color=INK)

    layers_x = [1.5, 4.5, 7.5]
    ys = [2.9, 1.5]
    node_labels = [
        ["x₁ = 1", "x₂ = 2"],
        ["z₁ = 0.7\na₁ = 0.7", "z₂ = 0.6\na₂ = 0.6"],
        ["z = 0.41\np = 0.655", "z = 0.23\np = 0.345"],
    ]
    colors = [GREY, BLUE, GREEN]
    for lx, labels, color in zip(layers_x, node_labels, colors):
        for ny, label in zip(ys, labels):
            ax.add_patch(Circle((lx, ny), 0.42, facecolor="white", edgecolor=color, lw=2))
            ax.text(lx, ny, label, ha="center", va="center", fontsize=8.5, color=INK)

    W1 = [[0.1, -0.2], [0.3, 0.4]]
    W2 = [[0.5, -0.5], [0.1, 0.2]]
    for (x0, x1, W) in [(layers_x[0], layers_x[1], W1), (layers_x[1], layers_x[2], W2)]:
        for i, y0 in enumerate(ys):
            for j, y1 in enumerate(ys):
                ax.plot([x0 + 0.42, x1 - 0.42], [y0, y1], color=GREY, lw=1, zorder=0)
                tx, ty = x0 + 0.42 + (x1 - x0 - 0.84) * (0.3 if i == j else 0.72), \
                         y0 + (y1 - y0) * (0.3 if i == j else 0.72)
                ax.text(tx, ty + 0.12, f"{W[i][j]:+.1f}", fontsize=8, color=BLUE, ha="center")

    ax.text(1.5, 0.55, "input", ha="center", fontsize=10, color=GREY)
    ax.text(4.5, 0.55, "hidden (ReLU)", ha="center", fontsize=10, color=BLUE)
    ax.text(7.5, 0.55, "output (softmax)", ha="center", fontsize=10, color=GREEN)

    ax.text(9.8, 3.4, "label y = [0, 1]", fontsize=10, color=INK, ha="center")
    ax.text(9.8, 2.9, "loss = -log 0.345\n       = 1.064", fontsize=10, color=RED, ha="center")
    ax.text(9.8, 2.0, "dL/dz = p - y\n= [0.655, -0.655]", fontsize=10, color=RED, ha="center")
    ax.text(9.8, 1.1, "gradients then flow\nback along the edges",
            fontsize=9, color=RED, ha="center", style="italic")
    fig.savefig(OUT / "worked_example.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_softmax_stability():
    z = np.linspace(-5, 5, 400)
    logits = np.stack([z, np.zeros_like(z)], axis=1)
    p_naive = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.plot(z, p_naive[:, 0], color=BLUE, lw=2, label="softmax([z, 0])_1 = 1/(1+e^{-z})")
    ax.axhline(0.5, color=GREY, lw=0.8, ls="--")
    ax.set_xlabel("logit z")
    ax.set_ylabel("probability of class 1")
    ax.set_title("Softmax squashes logits into probabilities")
    ax.legend(fontsize=8, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "softmax.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig_architecture()
    fig_chain_rule()
    fig_gradient_descent()
    fig_worked_example()
    fig_softmax_stability()
    print(f"wrote figures to {OUT}")
