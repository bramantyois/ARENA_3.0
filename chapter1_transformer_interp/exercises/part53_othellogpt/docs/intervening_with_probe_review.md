# Review: Intervening with the probe

## The big idea

The linear probe doesn't just *read* the model's internal board representation — it also lets you *write* to it. Because the probe found a meaningful direction in the residual stream, you can push the residual stream along that direction to change what the model believes about a specific square, and observe whether its move predictions respond correctly.

This is called a **causal intervention** — you're not just observing correlation, you're actively manipulating the model's internals and checking the downstream effect.

---

## Step 1 — Decompose the residual stream

For a given square, `my_probe[:, r, c]` gives a 512-dimensional direction $\vec{v}$ that encodes "mine vs theirs". Any residual stream vector at position `pos` can be decomposed as:

$$\text{resid}_{pos} = \alpha \cdot \hat{v} + \beta \cdot \vec{w}$$

- $\hat{v} = \frac{\vec{v}}{||\vec{v}||}$ — the **unit vector** (normalized probe direction)
- $\alpha = \text{resid}_{pos} \cdot \hat{v}$ — the scalar **projection** (how strongly the model believes "mine")
- $\beta \cdot \vec{w}$ — everything **orthogonal** to $\hat{v}$ (all other information, untouched)

---

## Step 2 — The `apply_scale` function

You want to flip the belief from $\alpha$ to $-\text{scale} \times \alpha$:

$$\text{resid}_{pos}^{\text{new}} = -\text{scale} \times \alpha \cdot \hat{v} + \beta \cdot \vec{w}$$

The net change needed is:

$$\Delta = (-\text{scale} \times \alpha) - \alpha = -(\text{scale} + 1) \times \alpha$$

So you subtract this from the residual:

$$\text{resid}_{pos} \mathrel{-}= (\text{scale} + 1) \cdot \alpha \cdot \hat{v}$$

Implementation:

```python
flip_dir_normed = flip_dir / flip_dir.norm()       # normalize → v̂
alpha = resid[:, pos, :] @ flip_dir_normed          # project → α, shape (batch,)
alpha = alpha.unsqueeze(-1)                         # shape (batch, 1) for broadcasting
resid[:, pos, :] -= (1 + scale) * alpha * flip_dir_normed
```

---

## Step 3 — The intervention in action

Flipping square **F4** from "mine → theirs" at layer 4 changed the model's predictions exactly as expected:

| Square | Before flip | After flip |
|--------|------------|-----------|
| `G4`   | legal ✅    | illegal ❌ |
| `D2`   | illegal ❌  | legal ✅   |

These are precisely the squares that *should* change when F4 switches color — verified by the `OthelloBoardState` ground truth. Other squares are mostly unaffected (at low scale values), which confirms the intervention is **surgical**.

---

## Key takeaways

1. **Linear representation is real and causal** — the model doesn't just correlate with the board state, it uses these directions to make downstream decisions.
2. **Scale matters** — too small and the flip isn't convincing; too large and you corrupt other information stored in the residual stream.
3. **The orthogonal component $\beta \vec{w}$ is preserved** — this is what makes the intervention clean. You change one belief without destroying everything else the model knows.

---

# Review: Reading off neuron weights

## The big idea

Having an interpretable set of directions in the residual stream (from the probe) means we can also interpret a neuron's internal behaviour directly — by projecting its weight vectors onto the probe directions.

Each MLP layer computes:

$$\text{MLP}(x) = f(x^\top W^{\text{in}}) \cdot W^{\text{out}}$$

which can be decomposed neuron by neuron as a sum:

$$\text{MLP}(x) = \sum_{n=0}^{d_{\text{mlp}}-1} f\!\left(x^\top W^{\text{in}}_{[:, n]}\right) \cdot W^{\text{out}}_{[n, :]}$$

where:
- $W^{\text{in}}_{[:, n]} \in \mathbb{R}^{d_{\text{model}}}$ is the **input weight vector** of neuron $n$ — a column of `W_in`
- $W^{\text{out}}_{[n, :]} \in \mathbb{R}^{d_{\text{model}}}$ is the **output weight vector** of neuron $n$ — a row of `W_out`
- $f$ is the activation function (ReLU/GELU)

In TransformerLens:
```python
model.W_in   # shape (n_layers, d_model, d_mlp)
model.W_out  # shape (n_layers, d_mlp, d_model)
```

So for neuron `n` at layer `l`:
- Input weight: `model.W_in[l, :, n]` → shape `(d_model,)`
- Output weight: `model.W_out[l, n, :]` → shape `(d_model,)`

---

## Projecting weights onto probe directions

To make the weights interpretable, we project them onto the normalized probe directions. Because both vectors are normalized to unit norm, the result is a **cosine similarity** — a value between -1 and 1:

$$\text{input score}[i, j] = \hat{W}^{\text{in}}_{[:, n]} \cdot \hat{p}_{[i,j]}$$

$$\text{output score}[i, j] = \hat{W}^{\text{out}}_{[n, :]} \cdot \hat{p}_{[i,j]}$$

where $\hat{p}_{[i,j]}$ is the normalized probe direction for square $(i, j)$, and $\hat{W}$ denotes a unit-normalized weight vector.

- A **high input score** for square $(i, j)$ means the neuron tends to fire when the residual stream encodes that square in the probe direction
- A **high output score** for square $(i, j)$ means the neuron writes in the direction of that probe when it fires

We normalize before projecting because we care about *alignment*, not scale (you could double the input magnitude and halve the output and get the same MLP result, ignoring biases).

---

## Case study: Neuron L5N1393

Looking at neuron 1393 in layer 5:

**Input weights (blank probe + my probe):** The neuron fires strongly when:
- `C0` is **blank**
- `D1` is **theirs**
- `E2` is **mine**

This corresponds exactly to the pattern needed for `C0` to be a legal move — placing there would flip `D1` (a diagonal chain).

**Output weights:** Tested via **direct logit attribution** (see below) — the neuron boosts the logit for `C0`, confirming the hypothesis.

---

## How much variance does the probe explain?

To test how much of a neuron's behaviour is captured by the probe, we measure what fraction of the (unit-norm) weight vector lies in the **span of all probe directions** — a quantity closely analogous to $R^2$ in regression.

### Step 1 — Build the combined probe matrix

We have two probe tensors, each of shape `(d_model=512, 8, 8)`. Flatten the board dimensions:

$$
M_{\text{my}} \in \mathbb{R}^{512 \times 64}, \quad M_{\text{blank}} \in \mathbb{R}^{512 \times 64}
$$

Then concatenate into a single matrix $P$:

$$
P = \begin{bmatrix} M_{\text{my}} \mid M_{\text{blank}} \end{bmatrix} \in \mathbb{R}^{512 \times 128}
$$

Each of the 128 columns is a probe direction for one cell under one concept (mine/theirs or blank). These columns are **not orthogonal** — nearby cells share redundant information.

### Step 2 — SVD to extract an orthonormal basis

Apply SVD to $P$:

$$
P = U \Sigma V^\top
$$

- $U \in \mathbb{R}^{512 \times 512}$ — orthonormal, columns span all of $\mathbb{R}^{512}$
- $\Sigma$ — diagonal, singular values $\sigma_1 \geq \sigma_2 \geq \dots \geq 0$
- $V^\top$ — how original probe columns combine into each principal direction (not used here)

The **first $r$ columns of $U$** (where $r = \text{rank}(P) \leq 128$) form an orthonormal basis for the **column space of $P$** — the probe subspace. The remaining columns of $U$ span its orthogonal complement (directions no probe touches at all).

In code:
```python
U, S, Vh = t.svd(t.cat([my_probe.reshape(cfg.d_model, 64), blank_probe.reshape(cfg.d_model, 64)], dim=1))
probe_space_basis = U[:, :-4]   # drop last 4 columns (see below)
```

### Step 3 — Why drop the last 4 columns?

The 4 center squares are **never blank** (they're always occupied at the start of the game), so their `blank_probe` vectors are effectively zero. SVD still produces $U$ columns for them (since $U$ must be square), but those columns correspond to near-zero singular values and are unconstrained by the data — they're numerical noise. Dropping them gives a clean basis for the meaningful probe subspace of dimension $\approx 124$.

### Step 4 — Measure variance explained

For a unit-norm weight vector $\hat{\mathbf{w}} \in \mathbb{R}^{512}$ (either $\hat{W}^{\text{in}}_{[:, n]}$ or $\hat{W}^{\text{out}}_{[n, :]}$), its orthogonal projection onto the probe subspace is:

$$
\hat{\mathbf{w}}_{\text{proj}} = U_{\text{probe}} \, U_{\text{probe}}^\top \, \hat{\mathbf{w}}
$$

The **fraction of variance explained** is the squared norm of this projection:

$$
\text{score} = \|\hat{\mathbf{w}}_{\text{proj}}\|^2 = \|U_{\text{probe}}^\top \, \hat{\mathbf{w}}\|^2 = \sum_{i} (\hat{\mathbf{w}} \cdot U_i)^2
$$

In code: `(w @ probe_space_basis).pow(2).sum()`

Since $\hat{\mathbf{w}}$ is unit norm, $\sum_{\text{all }i}(\hat{\mathbf{w}} \cdot U_i)^2 = 1$, so the score is bounded in $[0, 1]$ and directly interpretable as an $R^2$.

### Step 5 — The random baseline

The probe subspace has dimension $\approx 124$ out of $512$. By symmetry, a **uniformly random** unit vector in $\mathbb{R}^{512}$ has expected score:

$$
\mathbb{E}[\text{score}] = \frac{124}{512} \approx 0.24
$$

So any score significantly above 0.24 is meaningful — the neuron's weights are systematically aligned with board-state directions.

### Results for L5N1393

| Weight vector | Score | Interpretation |
|---|---|---|
| Input $\hat{W}^{\text{in}}_{[:, n]}$ | **high** (> 0.24) | Neuron fires in response to board-state features — blank/mine/theirs for specific cells |
| Output $\hat{W}^{\text{out}}_{[n, :]}$ | **lower** (~0.24) | Neuron writes toward move predictions (unembedding space), not back into the board-state representation |

This asymmetry is exactly what you'd expect for a "C0 is legal" detector: it *reads* from the board state (high input alignment) but *writes* to the output logits (low output alignment with the probe subspace).

---

## More neurons: Layer 4 "blankness" neurons

The top neurons in layer 4 (by activation standard deviation) all show striking alignment with a **single blank probe direction**. This makes sense:

- Being blank is a *necessary* condition for a square to be legal
- "Is this square blank?" is easy to compute — just check if it's been played yet — and can be done in a single layer
- These neurons detect blankness and write directly to the residual stream in a way that increases the corresponding move logit

This is confirmed by direct logit attribution (see below) — their output weights are highly cosine-similar with the unembedding column for the blank square they're detecting.

---

# Review: Direct Logit Attribution (DLA)

## The big idea

**Direct logit attribution** (DLA) measures how much a model component directly affects the final logits, *ignoring* effects that pass through intermediate components.

Since the logits are computed as:

$$\text{logits} = \text{resid\_final} \cdot W_U$$

and the residual stream is a sum of contributions from all components:

$$\text{resid\_final} = x_0 + \sum_{\ell, \text{comp}} \Delta_{\ell, \text{comp}}$$

the direct logit contribution of a single component (e.g., a neuron) is:

$$\text{DLA}_n = W^{\text{out}}_{[n,:]} \cdot W_U$$

This is a vector of shape `(d_vocab,)` — the direct effect of neuron $n$ firing (with unit activation) on each token logit.

## Computing DLA for a neuron

In practice, we compute the **cosine similarity** between the neuron's (normalized) output weights and the (normalized) unembedding columns:

$$\text{cos\_sim}[k] = \hat{W}^{\text{out}}_{[n,:]} \cdot \hat{W}_U^{[:,k]}$$

where $\hat{W}_U^{[:,k]}$ is the normalized unembedding column for token $k$.

Cosine similarity is preferred over raw dot product because:
- 1 is the theoretical maximum → easy to interpret
- The expected absolute value for two random vectors in $d$-dimensional space is $\approx \frac{1}{\sqrt{d}}$ (here $\frac{1}{\sqrt{512}} \approx 0.04$), giving a natural baseline for comparison

In code:
```python
w_out_normed = get_w_out(model, layer, neuron, normalize=True)  # shape (d_model,)
W_U_normalized = model.W_U[:, 1:] / model.W_U[:, 1:].norm(dim=0, keepdim=True)  # shape (d_model, 60)
cos_sim = w_out_normed @ W_U_normalized  # shape (60,)
```

The result is then plotted on an 8×8 board to see which squares the neuron is directly predicting as legal.

## Key finding

For **L5N1393**: the neuron directly boosts the logit for `C0` (and slightly `D1`), consistent with the hypothesis that it detects the `(C0==blank, D1==theirs, E2==mine)` pattern and predicts `C0` as a legal move.

For **top layer 4 neurons**: each neuron's DLA map matches its blank probe alignment exactly — neurons detecting blank squares directly write to raise the logit for those squares. This is a clean example of a modular circuit: *detect blankness → predict legality*.
