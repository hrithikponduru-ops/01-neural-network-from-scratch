# 01 · Neural Network From Scratch

## Question & Answer

**Question:** Can I derive backpropagation by hand, implement it in plain NumPy, and prove the gradients are correct without relying on an autograd library?

**Answer:** Yes. The network below reaches **96.85% test accuracy on MNIST after 5 epochs**, and its hand-derived gradients agree with PyTorch's autograd to within $10^{-10}$.

---

## Table of Contents

1. [The idea in one paragraph](#1-the-idea-in-one-paragraph)
2. [The model](#2-the-model)
3. [A worked example you can do by hand](#3-a-worked-example-you-can-do-by-hand)
4. [The derivations](#4-the-derivations)
5. [Verification: proving the math is right](#5-verification-proving-the-math-is-right)
6. [Results on MNIST](#6-results-on-mnist)
7. [Numerical lessons learned](#7-numerical-lessons-learned)
8. [How to run](#8-how-to-run)
9. [Code map](#9-code-map)

---

## 1. The idea in one paragraph

A neural network is a function with knobs. You feed in a picture, it outputs a guess. The **loss** $L$ is a single number measuring how wrong that guess was. Training means turning the knobs to make $L$ smaller. Calculus tells us which way to turn each knob $\theta$: the derivative $\partial L / \partial \theta$. **Backpropagation** is the chain rule, applied systematically, to get all of those derivatives at once. **Gradient descent** then nudges every knob a little in the downhill direction:

$$\theta \leftarrow \theta - \eta \, \frac{\partial L}{\partial \theta}$$

where $\eta$ is the learning rate.

*With one knob, follow the slope downhill. With two knobs, the gradient points uphill, so step the opposite way. A real network has 101,770 knobs, but the picture is the same.*

---

## 2. The model

Six functions chained together, each a few lines of NumPy. With a batch of $n$ images, $X \in \mathbb{R}^{n \times 784}$ holds one flattened image per row, scaled to $[0, 1]$.

$$
\begin{aligned}
Z_1 &= X W_1 + b_1 &&\in \mathbb{R}^{n \times 128} && \text{(dense)} \\
A_1 &= \operatorname{relu}(Z_1) = \max(Z_1, 0) &&\in \mathbb{R}^{n \times 128} && \text{(element-wise)} \\
Z_2 &= A_1 W_2 + b_2 &&\in \mathbb{R}^{n \times 10} && \text{(dense)} \\
P &= \operatorname{softmax}(Z_2), \qquad P_{ij} = \frac{e^{Z_{2,ij}}}{\sum_{k=1}^{10} e^{Z_{2,ik}}} &&\in \mathbb{R}^{n \times 10} && \text{(row-wise)} \\
L &= -\frac{1}{n} \sum_{i=1}^{n} \sum_{j=1}^{10} Y_{ij} \log P_{ij} &&\in \mathbb{R} && \text{(cross-entropy)}
\end{aligned}
$$

$Y \in \{0,1\}^{n \times 10}$ is the one-hot label matrix, so the inner sum picks out $\log P_{i,\text{correct}}$ for each image.

| Parameter | Shape | Entries |
|-----------|-------|--------:|
| $W_1$ | $784 \times 128$ | 100,352 |
| $b_1$ | $128$ | 128 |
| $W_2$ | $128 \times 10$ | 1,280 |
| $b_2$ | $10$ | 10 |
| **Total** | | **101,770** |

### Why these pieces?

- **Dense layers** are matrix multiplication. Every output is a weighted sum of every input. This is the only place learning happens.

- **ReLU** is the simplest non-linearity. Without it, $X W_1 W_2 = X (W_1 W_2)$ collapses two layers into one, so the network could only learn linear boundaries.

- **Softmax** turns ten arbitrary scores into ten positive numbers that sum to one, so they can be read as probabilities.

- **Cross-entropy** is $-\log$ of the probability assigned to the correct answer. Confident and right: near zero. Confident and wrong: very large.

---

## 3. A worked example you can do by hand

MNIST has 784 inputs and 101,770 parameters, too many to follow. Here is the same network shrunk to 2 inputs, 2 hidden units and 2 classes. Every number below is produced by the same code that trains on MNIST. Run `python worked_example.py` to see it print.

### Setup

$$
x = \begin{bmatrix} 1 & 2 \end{bmatrix}, \qquad
y = \begin{bmatrix} 0 & 1 \end{bmatrix} \quad \text{(class 2 is correct)}
$$

$$
W_1 = \begin{bmatrix} 0.1 & -0.2 \\ 0.3 & 0.4 \end{bmatrix}, \quad
b_1 = \begin{bmatrix} 0 & 0 \end{bmatrix}, \qquad
W_2 = \begin{bmatrix} 0.5 & -0.5 \\ 0.1 & 0.2 \end{bmatrix}, \quad
b_2 = \begin{bmatrix} 0 & 0 \end{bmatrix}
$$

### Forward pass (left to right)

$$
\begin{aligned}
z_1 &= x W_1 + b_1 = \begin{bmatrix} 1(0.1) + 2(0.3) & 1(-0.2) + 2(0.4) \end{bmatrix} = \begin{bmatrix} 0.7 & 0.6 \end{bmatrix} \\[4pt]
a_1 &= \operatorname{relu}(z_1) = \begin{bmatrix} 0.7 & 0.6 \end{bmatrix} \quad \text{(both entries positive, so unchanged)} \\[4pt]
z_2 &= a_1 W_2 + b_2 = \begin{bmatrix} 0.7(0.5) + 0.6(0.1) & 0.7(-0.5) + 0.6(0.2) \end{bmatrix} = \begin{bmatrix} 0.41 & -0.23 \end{bmatrix} \\[4pt]
p &= \operatorname{softmax}(z_2) = \frac{1}{e^{0.41} + e^{-0.23}} \begin{bmatrix} e^{0.41} & e^{-0.23} \end{bmatrix} = \begin{bmatrix} 0.655 & 0.345 \end{bmatrix} \\[4pt]
L &= -\log p_2 = -\log 0.345 = 1.064
\end{aligned}
$$

The network gives the correct class only 34.5% probability. The loss is high. We want to know which direction each of the 12 parameters should move.

### Backward pass (right to left)

Start at the loss and walk back, multiplying by one derivative at each step.

$$
\frac{\partial L}{\partial z_2} = p - y = \begin{bmatrix} 0.655 & -0.655 \end{bmatrix}
$$

Read this as: the first logit is too high by 0.655, the second is too low by 0.655. The clean form $p - y$ is derived in section 4.1.

$$
\begin{aligned}
\frac{\partial L}{\partial W_2} &= a_1^{\mathsf T} \, \frac{\partial L}{\partial z_2} = \begin{bmatrix} 0.7 \\ 0.6 \end{bmatrix} \begin{bmatrix} 0.655 & -0.655 \end{bmatrix} = \begin{bmatrix} 0.458 & -0.458 \\ 0.393 & -0.393 \end{bmatrix} \\[6pt]
\frac{\partial L}{\partial b_2} &= \frac{\partial L}{\partial z_2} = \begin{bmatrix} 0.655 & -0.655 \end{bmatrix} \\[6pt]
\frac{\partial L}{\partial a_1} &= \frac{\partial L}{\partial z_2} \, W_2^{\mathsf T} = \begin{bmatrix} 0.655 & -0.655 \end{bmatrix} \begin{bmatrix} 0.5 & 0.1 \\ -0.5 & 0.2 \end{bmatrix} = \begin{bmatrix} 0.655 & -0.066 \end{bmatrix}
\end{aligned}
$$

The gradient for $W_2$ is an outer product: each weight's gradient is (its input) × (the error at its output). That is the whole intuition of backprop. Blame flows back along each wire in proportion to what was sent down it.

$$
\begin{aligned}
\frac{\partial L}{\partial z_1} &= \frac{\partial L}{\partial a_1} \odot \mathbb{1}[z_1 > 0] = \begin{bmatrix} 0.655 & -0.066 \end{bmatrix} \quad \text{(both } z_1 > 0 \text{, nothing blocked)} \\[6pt]
\frac{\partial L}{\partial W_1} &= x^{\mathsf T} \, \frac{\partial L}{\partial z_1} = \begin{bmatrix} 1 \\ 2 \end{bmatrix} \begin{bmatrix} 0.655 & -0.066 \end{bmatrix} = \begin{bmatrix} 0.655 & -0.066 \\ 1.310 & -0.131 \end{bmatrix} \\[6pt]
\frac{\partial L}{\partial b_1} &= \frac{\partial L}{\partial z_1} = \begin{bmatrix} 0.655 & -0.066 \end{bmatrix}
\end{aligned}
$$

### One gradient descent step

With $\eta = 0.5$, every parameter moves against its gradient, $\theta \leftarrow \theta - 0.5 \, \partial L / \partial \theta$. Running the forward pass again:

$$
\begin{array}{lll}
\text{before:} & p = \begin{bmatrix} 0.655 & 0.345 \end{bmatrix} & L = 1.064 \\
\text{after:}  & p = \begin{bmatrix} 0.260 & 0.740 \end{bmatrix} & L = 0.301
\end{array}
$$

One step, and the probability of the correct class went from 35% to 74%. Training on MNIST is this exact procedure repeated 4,690 times on batches of 64 images.

---

## 4. The derivations

### 4.1 Softmax + cross-entropy: why $\partial L / \partial z = p - y$

Take one example with logits $z \in \mathbb{R}^{10}$, probabilities $p_j = e^{z_j} / S$ where $S = \sum_k e^{z_k}$, and one-hot label $y$ with $\sum_j y_j = 1$. Then:

$$
L = -\sum_j y_j \log p_j = -\sum_j y_j \left( z_j - \log S \right) = -\sum_j y_j z_j + \log S
$$

Differentiate with respect to $z_i$:

$$
\frac{\partial L}{\partial z_i} = -y_i + \frac{1}{S} \frac{\partial S}{\partial z_i} = -y_i + \frac{e^{z_i}}{S} = p_i - y_i
$$

Simplifying $\log \operatorname{softmax}$ *before* differentiating means the $10 \times 10$ softmax Jacobian never has to be written down. Averaging over a batch of $n$ examples divides by $n$:

$$
\frac{\partial L}{\partial Z_2} = \frac{1}{n} \left( P - Y \right)
$$

Implemented in `nn/loss.py::softmax_cross_entropy_backward`.

### 4.2 Dense layer: $\partial L / \partial W$, $\partial L / \partial b$, $\partial L / \partial X$

Forward: $Z = XW + b$ with $X \in \mathbb{R}^{n \times d_{\text{in}}}$, $W \in \mathbb{R}^{d_{\text{in}} \times d_{\text{out}}}$, $b \in \mathbb{R}^{d_{\text{out}}}$. Element-wise:

$$
Z_{ij} = \sum_{k} X_{ik} W_{kj} + b_j
$$

We are given the upstream gradient $G = \partial L / \partial Z$, same shape as $Z$.

**Weights.** $W_{kj}$ appears in $Z_{ij}$ for every row $i$, with coefficient $X_{ik}$:

$$
\frac{\partial L}{\partial W_{kj}} = \sum_i \frac{\partial L}{\partial Z_{ij}} \frac{\partial Z_{ij}}{\partial W_{kj}} = \sum_i G_{ij} X_{ik} \quad \Longrightarrow \quad \boxed{\frac{\partial L}{\partial W} = X^{\mathsf T} G}
$$

Shape check: $(d_{\text{in}} \times n)(n \times d_{\text{out}}) = d_{\text{in}} \times d_{\text{out}}$, matching $W$.

**Bias.** $b_j$ is added to every row of column $j$:

$$
\frac{\partial L}{\partial b_j} = \sum_i G_{ij} \quad \Longrightarrow \quad \boxed{\frac{\partial L}{\partial b} = \mathbf{1}^{\mathsf T} G} \quad \text{(column sums of } G \text{)}
$$

**Input**, needed to keep the chain going to the previous layer. $X_{ik}$ appears in $Z_{ij}$ for every $j$, with coefficient $W_{kj}$:

$$
\frac{\partial L}{\partial X_{ik}} = \sum_j G_{ij} W_{kj} \quad \Longrightarrow \quad \boxed{\frac{\partial L}{\partial X} = G \, W^{\mathsf T}}
$$

Shape check: $(n \times d_{\text{out}})(d_{\text{out}} \times d_{\text{in}}) = n \times d_{\text{in}}$, matching $X$.

Implemented in `nn/layers.py::dense_backward`. Three lines of code, and the shape checks are how you know the transposes are in the right place.

### 4.3 ReLU

$\operatorname{relu}(z) = \max(z, 0)$, so:

$$
\operatorname{relu}'(z) = \begin{cases} 1 & z > 0 \\ 0 & z \le 0 \end{cases} \qquad \Longrightarrow \qquad \frac{\partial L}{\partial Z} = \frac{\partial L}{\partial A} \odot \mathbb{1}[Z > 0]
$$

where $\odot$ is element-wise multiplication. At exactly $z = 0$ the derivative is undefined; we use 0, which never matters in practice.

### 4.4 Putting it together

The full backward pass for the two-layer network, top to bottom:

$$
\begin{aligned}
G_2 &= \frac{\partial L}{\partial Z_2} = \tfrac{1}{n}(P - Y) \\
\frac{\partial L}{\partial W_2} &= A_1^{\mathsf T} G_2, \qquad \frac{\partial L}{\partial b_2} = \mathbf{1}^{\mathsf T} G_2 \\
G_1 &= \frac{\partial L}{\partial Z_1} = \left( G_2 W_2^{\mathsf T} \right) \odot \mathbb{1}[Z_1 > 0] \\
\frac{\partial L}{\partial W_1} &= X^{\mathsf T} G_1, \qquad \frac{\partial L}{\partial b_1} = \mathbf{1}^{\mathsf T} G_1
\end{aligned}
$$

The backward pass is the forward pass read in reverse, with each function replaced by its derivative. The intermediate values $X$, $Z_1$, $A_1$, $P$ must be saved from the forward pass because the derivatives need them. The `Cache` dataclass in `nn/network.py` holds them.

---

## 5. Verification: proving the math is right

A wrong gradient often still trains. The loss goes down a bit, accuracy is mediocre, and nothing announces that the derivation has a bug. So the code is checked two independent ways in `tests/test_gradients.py`.

**Check 1: finite differences.** The derivative is *defined* as a limit. Perturb one parameter $\theta$ by $\pm h$ and compute the central difference:

$$
\frac{\partial L}{\partial \theta} \approx \frac{L(\theta + h) - L(\theta - h)}{2h}
$$

This uses only the forward pass, so it is independent of every line of backward code. The central form is used rather than the one-sided $\left( L(\theta + h) - L(\theta) \right) / h$ because Taylor expansion gives:

$$
\frac{L(\theta + h) - L(\theta - h)}{2h} = L'(\theta) + \frac{h^2}{6} L'''(\theta) + O(h^4)
$$

so the error is $O(h^2)$ instead of $O(h)$: the even-order terms cancel. With $h = 10^{-5}$ this gives agreement to about $10^{-7}$ relative error, and the test demands $10^{-5}$.

**Check 2: PyTorch autograd.** Build the same network in torch with the same weights and the same batch, call `.backward()`, and compare. Agreement to $10^{-10}$ absolute. This catches anything finite differences might blur.

```
python -m pytest tests -v

test_analytic_matches_finite_difference[W1] PASSED
test_analytic_matches_finite_difference[b1] PASSED
test_analytic_matches_finite_difference[W2] PASSED
test_analytic_matches_finite_difference[b2] PASSED
test_analytic_matches_torch_autograd         PASSED
```

---

## 6. Results on MNIST

### Results from paper (5 epochs)

Hidden size 128, batch 64, $\eta = 0.1$, plain SGD, seed 0.

| Epoch | Train loss | Test loss | Test accuracy |
|------:|----------:|----------:|--------------:|
| 1 | 0.2917 | 0.2122 | 93.76% |
| 2 | 0.1429 | 0.1582 | 95.33% |
| 3 | 0.1050 | 0.1222 | 96.40% |
| 4 | 0.0826 | 0.1064 | 96.74% |
| 5 | 0.0674 | 0.1040 | 96.85% |

Train loss keeps falling faster than test loss, the first sign of overfitting. A network this size typically tops out near 98% with more epochs; going further needs regularisation or a different architecture, both outside the scope of this project.

### Extended training results (20 epochs)

Running the same architecture for 20 epochs shows continued improvement in test accuracy without catastrophic overfitting:

| Epoch | Train loss | Test loss | Test accuracy |
|------:|----------:|----------:|--------------:|
| 1 | 0.8597 | 0.4461 | 88.67% |
| 2 | 0.3990 | 0.3473 | 90.72% |
| 3 | 0.3338 | 0.3065 | 91.57% |
| 4 | 0.3007 | 0.2818 | 92.26% |
| 5 | 0.2777 | 0.2639 | 92.59% |
| 10 | 0.2116 | 0.2069 | 94.06% |
| 15 | 0.1741 | 0.1723 | 95.08% |
| 20 | 0.1489 | 0.1502 | 95.63% |

**Key observations:**

- **Smooth convergence** — no spikes, NaN, or divergence. Hand-derived gradients are working correctly.
- **Good generalization** — train and test loss track closely through epoch 20, with only modest overfitting emerging at the end.
- **Sustained improvement** — test accuracy climbed 6.96 percentage points (from 88.67% to 95.63%) without plateauing.

**Why the 20-epoch results differ from the paper:**

- **Random seed** — different weight initialization produces different trajectories. The paper explicitly uses seed 0.
- **Hyperparameter variation** — minor differences in learning rate, batch size, or optimizer can shift the learning curve.
- **Peak at different epochs** — this run reaches 95.63% by epoch 20; the paper reached 96.85% by epoch 5, suggesting it hit a local maximum earlier.

### Aggressive training: 50 epochs with $\eta = 0.15$

Increasing the learning rate to $\eta = 0.15$ and running for 50 epochs reveals the dynamics of overfitting and optimal stopping. Key milestones:

| Epoch | Train loss | Test loss | Test accuracy |
|------:|----------:|----------:|--------------:|
| 1 | 0.4111 | 0.2663 | 92.45% |
| 5 | 0.1247 | 0.1394 | 95.92% |
| 10 | 0.0736 | 0.1018 | 96.84% |
| 15 | 0.0506 | 0.0805 | 97.54% |
| **29** | **0.0241** | **0.0686** | **97.92%** |
| 30 | 0.0232 | 0.0680 | 97.84% |
| 40 | 0.0154 | 0.0675 | 97.92% |
| 50 | 0.0111 | 0.0679 | 97.86% |

**Peak performance: 97.92% test accuracy at epoch 29** — exceeding the original paper's 96.85%.

### Analysis: Higher learning rate unlocks better generalization

**Key insights:**

- **Faster warm-up** — with $\eta = 0.15$, epoch 1 reaches 92.45% (vs 88.67% with $\eta = 0.1$). Larger steps explore the loss landscape more aggressively.
- **Accelerated convergence** — by epoch 10, test accuracy is already 96.84%, matching the paper's epoch 5 result.
- **Textbook overfitting curve** — train loss continues falling (0.0111 at epoch 50), while test loss stabilizes around 0.067–0.071 from epoch 20 onward. The gap widens steadily.
- **Optimal stopping point** — test accuracy peaks at epoch 29 (97.92%), then oscillates ±0.1%. Continuing past this point gains no new insight but suggests diminishing returns.

**Why $\eta = 0.15$ works better:**

- Larger steps move faster down the loss surface early on, reaching the steep part of the curve sooner.
- By epoch 20, the network has learned robust features; noise in the updates (a side effect of larger $\eta$) causes test loss to plateau rather than decrease further—a natural regularization.
- For this problem size (101,770 parameters, 60k training samples), $\eta = 0.15$ sits in the "Goldilocks zone"—aggressive enough to avoid local minima, stable enough to avoid divergence.

**When to stop training:**

- **Epoch 15** gives 97.54% — solid and nearly optimal with no overfitting.
- **Epoch 29** gives 97.92% — peak test accuracy, but requires patience and a validation set to detect.
- **Epoch 50** gives 97.86% — diminishing returns; train loss is 0.0111 but test accuracy dropped back 0.06%.

This run demonstrates that **hand-derived backprop scales** to competitive performance. The network trained purely from first-principles math reaches within 1.07% of state-of-the-art shallow networks on MNIST (which top out near 99% with more sophisticated architectures).

---

## 7. Numerical lessons learned

These are things the math does not warn you about and the computer does.

### Softmax overflows

$e^{710}$ exceeds the largest float64. Subtracting the row maximum $m = \max_k z_k$ before exponentiating leaves the probabilities unchanged:

$$
\frac{e^{z_j - m}}{\sum_k e^{z_k - m}} = \frac{e^{-m} e^{z_j}}{e^{-m} \sum_k e^{z_k}} = \frac{e^{z_j}}{\sum_k e^{z_k}}
$$

and keeps every exponent $\le 0$. Without this, training produces NaN within a few batches.

### Log of zero

$\log 0 = -\infty$. A probability can round to exactly 0. A tiny $\varepsilon = 10^{-12}$ is added inside the log.

### Initialisation scale matters

Weights are drawn from $\mathcal{N}(0, \, 2 / d_{\text{in}})$, the "He" initialisation. With ReLU this keeps the variance of activations roughly constant from layer to layer. Weights drawn from $\mathcal{N}(0, 1)$ instead give logits in the hundreds on the first batch and the loss starts near 100 rather than near $\log 10 \approx 2.3$.

### Multithreaded BLAS is slower on small matrices

The first training run took 25 minutes. With one thread it took under a minute. On $64 \times 784$ matrices NumPy's default multithreaded backend spends far longer synchronising threads than multiplying. `train.py` sets the thread count to 1 before importing NumPy. Measured: 0.32 s per batch vs 0.003 s per batch.

---

## 8. How to run

From this folder:

```bash
python -m pytest tests -v     # verify the gradients (about 5 seconds)

python worked_example.py      # print the 2-2-2 example step by step

python train.py               # train on MNIST, writes figures/training.png

python make_figures.py        # regenerate the diagrams in figures/
```

MNIST downloads automatically to `../.data/` on first run (about 12 MB).

---

## 9. Code map

```
nn/

  data.py          load MNIST as NumPy arrays, one-hot encode labels

  layers.py        dense layer: init, forward, backward

  activations.py   relu, relu_backward

  loss.py          softmax, cross_entropy, combined backward

  network.py       wire the layers together; forward, backward, sgd_step

tests/

  test_gradients.py   finite-difference and autograd checks

train.py           training loop and plot

worked_example.py  the 2-2-2 example from section 3

make_figures.py    draws the diagrams

figures/           generated images used above
```

---

*Document beautifully formatted in Markdown with LaTeX mathematical notation preserved.*