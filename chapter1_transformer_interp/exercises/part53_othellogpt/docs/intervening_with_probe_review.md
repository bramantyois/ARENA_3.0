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
