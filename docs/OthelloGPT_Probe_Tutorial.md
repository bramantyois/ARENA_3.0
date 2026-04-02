# OthelloGPT Linear Probe Tutorial

A comprehensive guide to training linear probes and performing mechanistic interpretability analysis on OthelloGPT, with detailed explanations of each step's purpose and interpretability significance.

---

## Table of Contents

1. [Introduction](#introduction)
2. [Getting Started](#getting-started)
3. [Section 1: Model Setup & Linear Probes](#section-1-model-setup--linear-probes)
4. [Section 2: Looking for Modular Circuits](#section-2-looking-for-modular-circuits)
5. [Section 3: Neuron Interpretability Deep Dive](#section-3-neuron-interpretability-deep-dive)
6. [Section 4: Training a Probe from Scratch](#section-4-training-a-probe-from-scratch)

---

## Introduction

### What is OthelloGPT?

OthelloGPT is an 8-layer autoregressive transformer trained to predict the next move in Othello games. Despite only needing to predict a single move, the model **spontaneously learns to compute the full board state** at each step - an emergent world representation.

### Why is this important for Interpretability?

This emergent board state representation makes OthelloGPT an ideal "laboratory" for studying mechanistic interpretability because:

1. **Clear ground truth**: We know exactly what the board state is at each move
2. **Linear probe works**: The board state can be read from activations using a simple linear classifier
3. **Circuit analysis possible**: We can trace how the model computes board state through layers
4. **Intervention is effective**: We can modify model behavior by intervening on probe directions

### Key Concepts

| Concept | Definition |
|---------|------------|
| **Linear Probe** | A classifier trained on residual stream activations to predict board state |
| **Residual Stream** | The sum of all layer outputs; carries information through the network |
| **Activation Patching** | Replacing activations from one forward pass with another to test causal hypotheses |
| **Direct Logit Attribution** | Measuring how a neuron's output weights directly affect output logits |
| **Probe Basis** | An interpretable coordinate system (blank/mine/theirs) for analyzing weights |

---

## Getting Started

### Setup Code

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple, Callable
from torchtyping import TensorType
from fancy_einsum import einsum
import einops
from transformer_lens import HookedTransformer, HookedTransformerConfig

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

### Loading the Model

```python
# Load the pre-trained OthelloGPT model
model = HookedTransformer.from_pretrained(
    "othello-gpt",
    fold_ln=True,           # Fold LayerNorm into weights
    center_writing_weights=True,  # Center writing weights
    center_unembed=True,    # Center unembedding
    refactor_act_hook_points=True,  # Refactor activations
)

# Model config
cfg = model.cfg
# d_model = 512 (model dimension)
# n_layers = 8
# n_heads = 8 (attention heads per layer)
# d_vocab = 61 (60 squares + pass)
```

**Why these preprocessing steps?**
- `fold_ln=True`: Simplifies analysis by incorporating LayerNorm into weights
- `center_*` options: Remove trivial biases, making weights more interpretable

### Model Verification

```python
# Create a sample input (first 10 moves of a game)
sample_input = torch.tensor([[46, 21, 19, 36, 24, 20, 34, 18, 48, 39]])

# Get model predictions
logits = model(sample_input)
logprobs = torch.log_softmax(logits, dim=-1)

# Display predictions
print(f"Input: {sample_input.tolist()}")
print(f"Logprobs at final position: {logprobs[0, -1, :10]}")
```

**What this achieves**: Verifies the model is working correctly by checking it produces reasonable move predictions.

**Interpretability significance**: Understanding baseline model behavior before analysis.

### Board Visualization

```python
def logits_to_board(logit_array):
    """
    Convert logits to board state tensor.
    
    Args:
        logit_array: Shape (batch, seq, vocab) or (seq, vocab)
    
    Returns:
        board_tensor: Shape (batch, seq, 8, 8)
    """
    temp_board_state = np.full((8, 8), -100)  # Start with very negative values
    temp_board_state.flatten()[utils.to_square([i for i in range(60)])] = logit_array[..., 1:]
    return temp_board_state
```

**Why this works**: The logit vector contains scores for all 61 tokens (pass + 60 squares). By placing them in the correct board positions, we can visualize which moves the model thinks are legal.

---

## Section 1: Model Setup & Linear Probes

### Learning Objectives

1. Understand the Othello-GPT model structure
2. Learn board visualization tools
3. Understand what probes are and how to use them
4. Learn about probe bases and linear interventions

---

### Step 1: Loading and Preparing Data

```python
# Load game data
# board_seqs_id: shape (100000, 60), range 1-60 (token IDs for squares)
# board_seqs_square: shape (100000, 60), range 0-63 (actual board indices)

games_id = board_seqs_id[:10000]  # Use subset
games_square = board_seqs_square[:10000]
```

**What this achieves**: We need game data to:
1. Run the model and cache activations
2. Get ground truth board states for probe training/evaluation
3. Create valid/corrupted pairs for activation patching

**Interpretability significance**: Quality of analysis depends on representative data. We filter out games with "pass" moves to keep the task simpler.

---

### Step 2: Caching Model Activations

```python
def cache_activations(model, games, num_games=50):
    """
    Cache all model activations for later analysis.
    
    Why cache? Running the model many times for different analyses is expensive.
    We cache once and reuse activations throughout the notebook.
    """
    cache_dict = {}
    
    def save_out(act, hook):
        cache_dict[hook.name] = act.detach().clone()
        return act
    
    # Get only focus games
    focus_games = games[:num_games]
    focus_games_id = focus_games
    
    # Register hooks on all residual stream positions
    handles = []
    for l in range(model.cfg.n_layers):
        handles.append(model.add_hook(f"resid_post_{l}", save_out))
        handles.append(model.add_hook(f"attn_out_{l}", save_out))
        handles.append(model.add_hook(f"mlp_out_{l}", save_out))
    
    # Run forward pass
    model(focus_games_id)
    
    # Remove hooks
    for h in handles:
        h.remove()
    
    return cache_dict
```

**What this achieves**: Stores activations at every layer for later probing and analysis.

**Interpretability significance**: The residual stream is where all information accumulates. By caching it at each layer, we can track how information (like board state) evolves through the network.

---

### Step 3: Understanding the Linear Probe

```python
# Pre-trained probe directions
# full_linear_probe shape: (3, 512, 8, 8, 3)
# Dimension 0: 3 modes (odd moves, even moves, both)
# Dimensions 2-3: 8x8 board positions
# Dimension 4: 3 classes (blank, theirs, mine)

# For simplicity, we'll use just one mode (odd moves, black to play)
linear_probe = full_linear_probe[0]  # Shape: (512, 8, 8, 3)
```

**What is a probe?**
A probe is a linear classifier trained on the model's internal activations to predict some external property (here: board state).

**How it works mathematically**:
```
probe_output = einsum(residual_stream, linear_probe, "d_model, d_model r c o -> r c o")
```

**Interpretability significance**: If a linear probe can accurately predict board state from activations, it means the information is **linearly accessible** - there exist directions in activation space that correspond to interpretable features.

---

### Step 4: Comparing Probe Directions (Cosine Similarity)

```python
# Get probe directions for odd and even moves
probe_odd = full_linear_probe[0]  # Black to play
probe_even = full_linear_probe[1]  # White to play

# Calculate cosine similarity between probe directions
def calculate_probe_similarity(probe_odd, probe_even):
    """
    Calculate how similar the probe directions are between odd and even moves.
    
    This tells us if the model represents board state similarly regardless of
    whose turn it is.
    """
    # Flatten and normalize
    probe_odd_flat = probe_odd.flatten(start_dim=1)  # (d_model, 192)
    probe_even_flat = probe_even.flatten(start_dim=1)  # (d_model, 192)
    
    # Cosine similarity for each direction
    similarity = F.cosine_similarity(probe_odd_flat.T, probe_even_flat.T, dim=1)
    return similarity
```

**What this achieves**: Tests whether odd and even move probes are similar, which would suggest the model uses a consistent representation.

**Expected result**: High similarity, meaning the model's internal representation doesn't drastically change based on turn parity.

**Interpretability significance**: Confirms we can use a simplified probe (averaged across modes) for analysis.

---

### Step 5: Changing Probe Basis

The original probe is trained in "black vs white" basis, but it's more natural to think in "mine vs theirs" terms (relative to the current player).

```python
def create_theirs_mine_probe(linear_probe):
    """
    Transform probe from (blank, black, white) basis to (blank, theirs, mine) basis.
    
    Math:
    - blank_new = blank_old - (black + white)/2  (remove player-specific component)
    - theirs_new = white - black  (for odd moves; reversed for even)
    - mine_new = black - white    (for odd moves; reversed for even)
    
    Simplified version (for odd moves):
    - blank_probe = linear_probe[..., 0] - linear_probe[..., 1] * 0.5 - linear_probe[..., 2] * 0.5
    - my_probe = linear_probe[..., 2] - linear_probe[..., 1]
    """
    # blank: direction that distinguishes blank from filled
    blank_probe = linear_probe[..., 0] - linear_probe[..., 1] * 0.5 - linear_probe[..., 2] * 0.5
    
    # my: direction that distinguishes mine from theirs
    my_probe = linear_probe[..., 2] - linear_probe[..., 1]
    
    return blank_probe, my_probe

blank_probe, my_probe = create_theirs_mine_probe(linear_probe)
```

**What this achieves**: Creates interpretable directions in activation space:
- `blank_probe`: Points in direction that indicates "blankness"
- `my_probe`: Points in direction that indicates "my color"

**Interpretability significance**: This is the key to all subsequent analysis. Once we have these directions:
1. We can **read** what the model thinks about any square
2. We can **intervene** to flip the model's belief about any square
3. We can **trace** which layers compute which information

---

### Step 6: Applying the Probe

```python
def apply_probe(activation, probe_direction):
    """
    Apply probe to residual stream activation.
    
    Args:
        activation: Shape (batch, seq, d_model)
        probe_direction: Shape (d_model, rows, cols)
    
    Returns:
        probe_scores: Shape (batch, seq, rows, cols)
    """
    # Project activation onto probe direction
    probe_scores = einsum(activation, probe_direction, "b s d, d r c -> b s r c")
    return probe_scores

# Example: Apply to move 29 in game 0
layer = 6
game_index = 0
move = 29

residual_stream = focus_cache["resid_post", layer][game_index, move]  # (d_model,)
probe_output = einsum(residual_stream, my_probe, "d_model, d_model r c -> r c")
```

**What this achieves**: Converts raw activations into interpretable scores for each board position.

**Interpretability significance**: We can now "read off" what the model represents at any layer and any position. High score on `my_probe` at position (r,c) means the model thinks that square is "mine".

---

### Step 7: Computing Probe Accuracy

```python
def calculate_probe_accuracy(cache, probe, true_states, layer):
    """
    Calculate how accurately the probe predicts true board state.
    
    Args:
        cache: Activation cache
        probe: Probe direction tensor
        true_states: Ground truth board states (0=blank, -1=theirs, 1=mine)
        layer: Layer index to analyze
    
    Returns:
        accuracy: Fraction of correct predictions
    """
    # Get activations at this layer
    activations = cache["resid_post", layer]  # (batch, seq, d_model)
    
    # Apply probe
    probe_scores = apply_probe(activations, probe)  # (batch, seq, 8, 8)
    
    # Get predicted class for each position
    predictions = probe_scores.argmax(dim=-1)  # (batch, seq, 8, 8)
    
    # Compare to true state
    correct = (predictions == true_states).float()
    accuracy = correct.mean()
    
    return accuracy
```

**What this achieves**: Quantifies how well the probe captures the board state representation.

**Expected result**: High accuracy (>90%) by layer 6-7, indicating the model has fully computed board state by this point.

**Interpretability significance**: Validates that our probe is a faithful representation of what the model "knows" about the board.

---

### Step 8: Linear Intervention

```python
def apply_scale(
    resid: TensorType["batch", "seq", "d_model"],
    flip_dir: TensorType["d_model"],
    scale: float,
    pos: int,
) -> TensorType["batch", "seq", "d_model"]:
    """
    Intervene on the residual stream by scaling a specific direction.
    
    This is the key operation for causal interventions.
    
    Args:
        resid: Residual stream activations
        flip_dir: Direction in activation space (e.g., "mine - theirs")
        scale: How much to translate by (0 = no change, 1 = flip, >1 = over-correct)
        pos: Which sequence position to modify
    
    Returns:
        Modified residual stream
    """
    # Normalize the flip direction
    flip_dir_normed = flip_dir / flip_dir.norm()
    
    # Calculate current projection onto flip direction
    alpha = resid[0, pos] @ flip_dir_normed
    
    # Translate: move the activation by (scale + 1) times the projection
    # This effectively "flips" the sign when scale = 1
    resid[0, pos] -= (scale + 1) * alpha * flip_dir_normed
    
    return resid

# Example usage:
flip_dir = my_probe[:, cell_r, cell_c]  # Direction for specific square
layer = 4
scales = [0, 1, 2, 4, 8, 16]

for scale in scales:
    def flip_hook(resid, hook):
        return apply_scale(resid, flip_dir, scale, move)
    
    # Run model with intervention
    flipped_logits = model.run_with_hooks(
        input_games,
        fwd_hooks=[(f"resid_post_{layer}", flip_hook)]
    )
```

**What this achieves**: Modifies the model's internal representation to flip its belief about a square, then observes how predictions change.

**Example**: If we intervene to flip square F4 from "black" to "white":
- The model should now predict different legal moves
- Specifically, moves that were illegal become legal and vice versa

**Interpretability significance**: This is **causal validation** of our probe:
- If the probe direction truly represents "my color", then flipping it should causally change predictions in the expected way
- This confirms the representation is not just correlational but **causally involved** in the model's reasoning

---

## Section 2: Looking for Modular Circuits

### Learning Objectives

1. Use probes across multiple layers
2. Identify which layers contribute which information
3. Apply activation patching to test causal hypotheses
4. Understand how neurons compute features

---

### Step 1: Probing Across Layers

```python
def calculate_accumulated_probe_score(
    cache: ActivationCache,
    probe: TensorType["d_model", "rows", "cols"],
    layer: int,
    game_index: int,
    move: int,
) -> TensorType["layers", "rows", "cols"]:
    """
    Calculate probe score at each accumulated layer.
    
    The residual stream accumulates contributions from all previous layers.
    This shows us at which layer each piece of information appears.
    
    Args:
        cache: Activation cache
        probe: Probe direction
        layer: Max layer to compute up to
        game_index, move: Specific position in game
    
    Returns:
        Probe scores for layers 0 through layer (inclusive)
    """
    # Stack activations at each layer up to `layer`
    residual_streams = torch.stack([
        cache["resid_post", l][game_index, move]  # (d_model,)
        for l in range(layer + 1)
    ])  # (layer + 1, d_model)
    
    # Project each onto probe direction
    scores = einsum(
        residual_streams,
        probe,
        "layer d_model, d_model r c -> layer r c"
    )
    
    return scores
```

**What this achieves**: Shows how the board state representation evolves through layers.

**Expected result**: Early layers show partial information, later layers show complete board state.

**Interpretability significance**: Identifies which layers are "responsible" for computing different aspects of the board state.

---

### Step 2: Decomposing Layer Contributions (Attention vs MLP)

```python
def calculate_attn_and_mlp_probe_score_contributions(
    cache: ActivationCache,
    probe: TensorType["d_model", "rows", "cols"],
    layer: int,
    game_index: int,
    move: int,
) -> tuple[
    TensorType["layers", "rows", "cols"],
    TensorType["layers", "rows", "cols"]
]:
    """
    Decompose the contribution of attention vs MLP layers to the probe score.
    
    This tells us which component (attention or MLP) is responsible for computing
    different features of the board state.
    
    Returns:
        attn_contributions: Cumulative contribution from attention layers
        mlp_contributions: Cumulative contribution from MLP layers
    """
    # Stack attention outputs at each layer
    attn_outputs = torch.stack([
        cache["attn_out", l][game_index, move]  # (d_model,)
        for l in range(layer + 1)
    ])
    
    # Stack MLP outputs at each layer
    mlp_outputs = torch.stack([
        cache["mlp_out", l][game_index, move]  # (d_model,)
        for l in range(layer + 1)
    ])
    
    # Project onto probe direction
    attn_contributions = einsum(
        attn_outputs,
        probe,
        "layers d_model, d_model r c -> layers r c"
    )
    
    mlp_contributions = einsum(
        mlp_outputs,
        probe,
        "layers d_model, d_model r c -> layers r c"
    )
    
    return attn_contributions, mlp_contributions
```

**What this achieves**: Separates the contributions of attention mechanisms vs MLP computation.

**Expected result**: 
- MLP layers: Handle local features (e.g., "this square was just taken")
- Attention layers: Handle long-range dependencies (e.g., tracking the full board)

**Interpretability significance**: Reveals the **modular structure** of the circuit:
- Attention heads likely handle move history and board state tracking
- MLP neurons likely encode specific patterns or features

---

### Step 3: Reading Neuron Weights in Probe Basis

```python
def get_w_in(
    model: HookedTransformer,
    layer: int,
    neuron: int,
    normalize: bool = False,
) -> TensorType["d_model"]:
    """
    Get input weights for a specific MLP neuron.
    
    Input weights show what the neuron reads from the residual stream.
    High positive weight = neuron activates when that dimension is positive.
    High negative weight = neuron activates when that dimension is negative.
    """
    w_in = model.W_in[layer, :, neuron].detach().clone()  # (d_model,)
    if normalize:
        w_in = w_in / w_in.norm()
    return w_in


def get_w_out(
    model: HookedTransformer,
    layer: int,
    neuron: int,
    normalize: bool = False,
) -> TensorType["d_model"]:
    """
    Get output weights for a specific MLP neuron.
    
    Output weights show how the neuron affects downstream computations.
    High positive weight = neuron increases that dimension in residual stream.
    """
    w_out = model.W_out[layer, neuron, :].detach().clone()  # (d_model,)
    if normalize:
        w_out = w_out / w_out.norm()
    return w_out


def calculate_neuron_input_weights(
    model: HookedTransformer,
    probe: TensorType["d_model", "rows", "cols"],
    layer: int,
    neuron: int,
) -> TensorType["rows", "cols"]:
    """
    Project neuron input weights onto probe basis.
    
    This tells us what the neuron is sensitive to (in terms of board positions).
    """
    w_in = get_w_in(model, layer, neuron, normalize=True)
    
    # Project onto each probe direction
    weights_in_probe_basis = einsum(
        w_in,
        probe,
        "d_model, d_model r c -> r c"
    )
    
    return weights_in_probe_basis
```

**What this achieves**: Translates raw weight vectors into interpretable features.

**Example interpretation**:
If a neuron's input weights project strongly to `blank_probe[2, 0]`, it means the neuron is sensitive to whether square C0 is blank.

**Interpretability significance**: This is the key to **neuron interpretability**:
- Input weights tell us what conditions trigger the neuron
- Output weights tell us what effect the neuron has

---

### Step 4: Forming and Testing Hypotheses

```python
# Example: Analyzing neuron 1393 in layer 5
layer = 5
neuron = 1393

# Get input weights in probe basis
w_in_blank = calculate_neuron_input_weights(model, blank_probe, layer, neuron)
w_in_my = calculate_neuron_input_weights(model, my_probe, layer, neuron)

# Hypothesis: neuron fires when (C0==BLANK) & (D1==THEIRS) & (E2==MINE)
# Check: does w_in_blank[2,0] have high positive value?
#        does w_in_my[3,1] have high negative value?
#        does w_in_my[4,2] have high positive value?

# Test with output weights
w_out = get_w_out(model, layer, neuron, normalize=True)
W_U_normalized = model.W_U[:, 1:] / model.W_U[:, 1:].norm(dim=0, keepdim=True)

# Project output weights onto unembedding directions
cos_sim = w_out @ W_U_normalized  # Cosine similarity with each logit direction
```

**What this achieves**: Forms hypothesis from input weights, tests with output weights.

**Example hypothesis chain**:
1. Input weights suggest: "detects C0 blank, D1 theirs, E2 mine"
2. Output weights show: "boosts C0 logit"
3. Combined interpretation: "detects ladder pattern, predicts C0 legal"

**Interpretability significance**: This is a **complete circuit analysis**:
- We understand what triggers the neuron
- We understand what effect the neuron has
- We can predict the neuron's behavior on new inputs

---

### Step 5: Variance Explained by Probe

```python
def variance_explained_by_probe(
    model: HookedTransformer,
    probe_blank,
    probe_my,
    layer: int,
    neuron: int,
) -> tuple[float, float]:
    """
    Calculate what fraction of a neuron's weights lie in the probe subspace.
    
    If a neuron's weights are mostly in the probe subspace, it means the
    probe captures most of what that neuron does.
    """
    w_in = get_w_in(model, layer, neuron, normalize=True)
    w_out = get_w_out(model, layer, neuron, normalize=True)
    
    # Create basis for probe space
    # Concatenate all probe directions, remove center 4 (never blank)
    probe_concat = torch.cat([
        probe_blank.reshape(-1),
        probe_my.reshape(-1)
    ], dim=0)  # (d_model, 128)
    
    U, S, Vh = torch.svd(probe_concat.T)
    probe_basis = U[:, :-4]  # Remove center 4 squares
    
    # Calculate fraction of variance (squared projection norm)
    frac_in = ((w_in @ probe_basis).pow(2).sum()).item()
    frac_out = ((w_out @ probe_basis).pow(2).sum()).item()
    
    return frac_in, frac_out
```

**What this achieves**: Quantifies how well the probe captures neuron behavior.

**Expected result**: 
- Input weights: ~60-70% in probe space (neurons read mostly from probe dimensions)
- Output weights: ~10-20% in probe space (neurons also affect other things)

**Interpretability significance**: Validates that the probe is a **good basis** for analysis. If probes explained little variance, our interpretations would be incomplete.

---

### Step 6: Finding Interesting Neurons

```python
def find_top_neurons_by_std(
    cache: ActivationCache,
    layer: int,
    top_k: int = 10,
) -> TensorType["top_k"]:
    """
    Find neurons with highest activation standard deviation.
    
    Rationale: Neurons that activate rarely but strongly (high std) are likely
    detecting specific features, rather than being uniformly active.
    """
    activations = cache["post", layer]  # (batch, seq, d_mlp)
    
    # Compute std across all positions
    std_per_neuron = activations.std(dim=[0, 1])  # (d_mlp,)
    
    # Get top-k indices
    top_neuron_indices = std_per_neuron.argsort(descending=True)[:top_k]
    
    return top_neuron_indices
```

**What this achieves**: Identifies neurons worth investigating further.

**Why std dev**: Neurons with high std are likely **sparse detectors** - they activate strongly for specific patterns and weakly otherwise.

**Interpretability significance**: Focuses analysis effort on the most interpretable neurons, avoiding "average" neurons that don't do much interesting computation.

---

### Step 7: Activation Patching Setup

```python
# Create clean and corrupted games
# Corruption: flip a single move (e.g., change D3 from black to white)

# Get model outputs on both
original_cache = cache_activations(model, [clean_game])
corrupted_cache = cache_activations(model, [corrupted_game])
original_logits = model(clean_game)
corrupted_logits = model(corrupted_game)
```

**What this achieves**: Creates a controlled comparison where only one square differs.

**Why this matters**: To test if a layer is **causally necessary** for computing something, we need to see what happens when we corrupt information at that layer.

---

### Step 8: Creating the Patching Metric

```python
def patching_metric(
    patched_logits: TensorType["batch", "seq", "d_vocab"],
    target_square: str = "F0",
    original_log_probs: TensorType,
    corrupted_log_probs: TensorType,
) -> float:
    """
    Quantify how well patching restores performance.
    
    Metric calibrated so that:
    - 0 = same as corrupted (patching did nothing)
    - 1 = same as original (patching fully restored)
    
    Args:
        patched_logits: Model output with patched activations
        target_square: The square we're measuring (should be legal in original)
    
    Returns:
        Normalized metric value
    """
    patched_log_probs = torch.log_softmax(patched_logits, dim=-1)
    
    target_idx = utils.label_to_id(target_square)
    
    # Get log prob for target square
    patched_log_prob = patched_log_probs[0, -1, target_idx]
    
    # Normalize
    original_log_prob = original_log_probs[0, -1, target_idx]
    corrupted_log_prob = corrupted_log_probs[0, -1, target_idx]
    
    metric = (patched_log_prob - corrupted_log_prob) / (original_log_prob - corrupted_log_prob)
    
    return metric.item()
```

**What this achieves**: Quantifies the success of patching.

**Why calibration matters**: Raw log prob differences are hard to interpret. Normalization gives us a clear 0-1 scale.

**Interpretability significance**: Allows us to **rank layers** by their causal importance.

---

### Step 9: Running the Patching Experiment

```python
def patch_at_layer(
    model: HookedTransformer,
    corrupted_input: TensorType,
    clean_cache: ActivationCache,
    layer: int,
    activation_type: str = "resid_post",
) -> float:
    """
    Patch activations at a specific layer and measure effect.
    
    Args:
        model: The model
        corrupted_input: Corrupted game input
        clean_cache: Cached clean activations
        layer: Layer to patch
        activation_type: "attn_out", "mlp_out", or "resid_post"
    
    Returns:
        Patching metric value
    """
    def patch_hook(activation, hook):
        # Replace activation at final position with clean version
        activation[0, -1, :] = clean_cache[hook.name][0, -1, :]
        return activation
    
    # Run model with patching hook
    patched_logits = model.run_with_hooks(
        corrupted_input,
        fwd_hooks=[(f"{activation_type}_{layer}", patch_hook)]
    )
    
    # Measure effect
    metric = patching_metric(patched_logits, ...)
    
    return metric
```

**What this achieves**: Measures causal importance of each layer.

**Expected result**: 
- MLP0: Important (early board state computation)
- MLP5, MLP6: Important (refining board state)
- Attn7: Important (final move prediction)
- Other layers: Less important

**Interpretability significance**: Identifies the **causal circuit** for computing board state:
- Shows which layers are necessary vs. redundant
- Validates our understanding from the probe analysis

---

## Section 3: Neuron Interpretability Deep Dive

### Learning Objectives

1. Apply direct logit attribution
2. Use SVD to assess probe coverage
3. Create and interpret max-activating datasets
4. Validate hypotheses with spectrum plots

---

### Step 1: Direct Logit Attribution

```python
def direct_logit_attribution(
    model: HookedTransformer,
    layer: int,
    neuron: int,
) -> TensorType["rows", "cols"]:
    """
    Compute how a neuron's output weights directly affect output logits.
    
    This tells us the neuron's direct effect on predictions, without
    going through intermediate layers.
    
    Formula: neuron_output_weights @ unembedding_matrix
    """
    w_out = get_w_out(model, layer, neuron, normalize=False)  # (d_model,)
    
    # Project onto unembedding directions
    logit_effects = w_out @ model.W_U[:, 1:]  # (60,)
    
    # Reshape to board
    board_effects = torch.zeros(8, 8)
    board_effects.flatten()[ALL_SQUARES] = logit_effects
    
    return board_effects
```

**What this achieves**: Shows which logits the neuron directly influences.

**Expected pattern**: For a neuron detecting "C0 should be legal", we see high positive effect on C0 logit.

**Interpretability significance**: Direct logit attribution is the **gold standard** for understanding neuron function:
- Input weights tell us what the neuron detects
- Output weights tell us what it affects
- Combined, they give a complete picture

---

### Step 2: Variance Explained by Unembedding

```python
def variance_explained_by_unembedding(
    model: HookedTransformer,
    layer: int,
    neuron: int,
) -> float:
    """
    Calculate what fraction of a neuron's output weight variance is
    explained by the unembedding subspace.
    
    High value = neuron directly affects predictions in interpretable ways.
    """
    w_out = get_w_out(model, layer, neuron, normalize=True)
    
    # SVD of unembedding matrix
    U, S, Vh = torch.svd(model.W_U[:, 1:])
    
    # Projection onto unembedding space
    projection = w_out @ U
    variance_captured = (projection.norm().item() ** 2)
    
    return variance_captured
```

**What this achieves**: Quantifies how much of a neuron's output goes directly to logits.

**Expected result**: Varies by neuron. Some neurons may have output weights mostly in unembedding direction, others may feed into hidden layers.

**Interpretability significance**: Validates that our interpretation (from direct logit attribution) captures the neuron's primary function.

---

### Step 3: Max-Activating Datasets

```python
def find_max_activating_games(
    cache: ActivationCache,
    layer: int,
    neuron: int,
    quantile: float = 0.99,
) -> tuple[list, TensorType]:
    """
    Find games/moves where a neuron has highest activations.
    
    This gives us concrete examples to interpret what triggers the neuron.
    """
    activations = cache["post", layer][:, :, neuron]  # (batch, seq)
    
    # Find top activations
    threshold = activations.quantile(quantile)
    top_mask = activations > threshold
    
    # Extract corresponding board states
    board_states = cache["board_state"]  # Need to cache this separately
    top_board_states = board_states[top_mask]
    
    return top_board_states, activations[top_mask]
```

**What this achieves**: Shows concrete examples where the neuron fires strongly.

**Expected pattern**: For a "C0 blank, D1 theirs, E2 mine" detector, all top activations should have this pattern.

**Interpretability significance**: Provides **empirical validation** of our weight-based hypothesis.

**Caveat**: Can be misleading if the dataset has systematic biases. Always verify with multiple analyses.

---

### Step 4: Spectrum Plots

```python
def make_spectrum_plot(
    neuron_acts: TensorType["n_observations"],
    board_states: TensorType["n_observations", "rows", "cols"],
    label_conditions: Callable,
) -> plot:
    """
    Create histogram of neuron activations, colored by whether
    the hypothesis conditions are met.
    
    A good hypothesis produces a bimodal spectrum:
    - High activations when conditions met
    - Low activations when conditions not met
    """
    labels = label_conditions(board_states)  # Boolean array
    
    # Plot histogram
    fig = px.histogram(
        df={"acts": neuron_acts.tolist(), "label": labels.tolist()},
        x="acts",
        color="label",
        barmode="overlay"
    )
    
    return fig
```

**What this achieves**: Validates hypothesis across full distribution.

**Expected pattern**: Bimodal distribution - high activations when hypothesis true, low otherwise.

**Interpretability significance**: Spectrum plots are the **most rigorous validation** of neuron hypotheses:
- They show the neuron's behavior across all inputs
- They reveal edge cases and failure modes
- They confirm the feature is truly sparse and interpretable

---

## Section 4: Training a Probe from Scratch

### Learning Objectives

1. Set up probe training pipeline
2. Understand training hyperparameters and their effects
3. Log and evaluate probe performance
4. Train multiple probes simultaneously

---

### Step 1: Training Configuration

```python
@dataclass
class ProbeTrainingArgs:
    """Configuration for probe training."""
    
    # Which layer and positions to train on
    layer: int = 6        # Usually layer 6 has complete board state
    pos_start: int = 5    # Skip early moves (board not fully populated)
    pos_end: int = -5     # Skip final moves (game ending)
    
    # Board dimensions
    options: int = 3      # blank, theirs, mine
    rows: int = 8
    cols: int = 8
    
    # Training hyperparameters
    epochs: int = 3
    num_games: int = 10_000
    batch_size: int = 64
    learning_rate: float = 1e-3
    
    # Probe mode: 1 = odd moves only, 3 = all modes
    modes: int = 1
```

**What this achieves**: Centralizes all training parameters.

**Key considerations**:
- `layer`: Choose layer where board state is fully computed (typically 6-7)
- `pos_start/end`: Avoid edge cases where board state is ambiguous
- `epochs`: Probes train quickly; too many epochs causes overfitting
- `learning_rate`: Start with 1e-3, may need adjustment

---

### Step 2: Data Preparation

```python
def prepare_probe_training_data(
    games_id: TensorType["n_games", 60],
    games_square: TensorType["n_games", 60],
    args: ProbeTrainingArgs,
) -> tuple[TensorType, TensorType]:
    """
    Prepare activations and labels for probe training.
    
    Returns:
        activations: (n_samples, d_model)
        labels: (n_samples, 8, 8)
    """
    # Slice games to get only the positions we care about
    games = games_id[:, args.pos_start:args.pos_end]
    labels = games_square[:, args.pos_start:args.pos_end]
    
    # Get activations from model
    activations = cache_activations(model, games)
    activations = activations["resid_post", args.layer]  # (batch, seq, d_model)
    
    # Flatten batch and sequence dimensions
    activations = activations.reshape(-1, model.cfg.d_model)
    labels = labels.reshape(-1, 8, 8)
    
    return activations, labels
```

**What this achieves**: Creates training dataset from raw games.

**Key transformations**:
- Slice to relevant positions
- Get activations at target layer
- Flatten batch and sequence for standard training loop

---

### Step 3: Training Loop

```python
class LinearProbeTrainer:
    def __init__(self, model: HookedTransformer, args: ProbeTrainingArgs):
        self.model = model
        self.args = args
        
        # Initialize probe
        self.probe = nn.Linear(
            model.cfg.d_model,
            args.options * args.rows * args.cols
        )
        self.optimizer = torch.optim.Adam(self.probe.parameters(), lr=args.learning_rate)
        
    def training_step(self, activations: TensorType, labels: TensorType) -> float:
        """Single training step."""
        self.optimizer.zero_grad()
        
        # Forward pass
        predictions = self.probe(activations)  # (batch, 3*8*8)
        predictions = predictions.view(-1, 8, 8, 3)  # (batch, 8, 8, 3)
        
        # Calculate loss
        loss = F.cross_entropy(
            predictions.permute(0, 3, 1, 2),  # (batch, 3, 8, 8)
            labels  # (batch, 8, 8)
        )
        
        # Backward pass
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def train(self, games_id, games_square) -> dict:
        """Full training loop with evaluation."""
        # Prepare data
        activations, labels = prepare_probe_training_data(
            games_id, games_square, self.args
        )
        
        # Create dataloader
        dataset = TensorDataset(activations, labels)
        dataloader = DataLoader(dataset, batch_size=self.args.batch_size, shuffle=True)
        
        # Training loop
        history = {"loss": [], "accuracy": []}
        
        for epoch in range(self.args.epochs):
            # Training
            self.probe.train()
            for batch_activations, batch_labels in dataloader:
                loss = self.training_step(batch_activations, batch_labels)
                history["loss"].append(loss)
            
            # Evaluation
            self.probe.eval()
            with torch.no_grad():
                predictions = self.probe(activations)
                predictions = predictions.argmax(dim=-1)
                accuracy = (predictions == labels).float().mean()
                history["accuracy"].append(accuracy.item())
        
        return history
```

**What this achieves**: Complete probe training pipeline with evaluation.

**Key components**:
- Cross-entropy loss for 3-class classification at each position
- Standard Adam optimizer
- Evaluation loop for monitoring progress

---

### Step 4: Multi-Mode Training (Bonus)

```python
@dataclass
class MultiProbeTrainingArgs(ProbeTrainingArgs):
    """Training args for multiple probe modes simultaneously."""
    modes: int = 3  # odd, even, all
    
    def setup_linear_probe(self, model: HookedTransformer):
        """Create separate probes for each mode."""
        probes = nn.Parameter(torch.randn(
            self.modes,
            model.cfg.d_model,
            self.rows,
            self.cols,
            self.options
        ) / np.sqrt(model.cfg.d_model))
        return probes
```

**What this achieves**: Trains separate probes for different move types.

**Why separate probes**: The model might represent board state differently for odd vs even moves (different player's turn).

**Expected result**: All three probes should have similar performance if representation is turn-invariant.

---

## Quick Reference

### Key Functions Summary

| Function | Purpose | Key Output |
|----------|---------|------------|
| `cache_activations()` | Store model activations | Dictionary of (layer, activation) |
| `create_theirs_mine_probe()` | Transform probe basis | `blank_probe`, `my_probe` |
| `apply_probe()` | Read activations | Board scores (8x8) |
| `apply_scale()` | Intervene on activations | Modified residual stream |
| `calculate_accumulated_probe_score()` | Track info through layers | Scores per layer |
| `calculate_attn_and_mlp_probe_score_contributions()` | Decompose contributions | Attn/MLP scores per layer |
| `get_w_in()` / `get_w_out()` | Access neuron weights | Weight vectors |
| `calculate_neuron_input_weights()` | Project weights to probe basis | Interpretability map |
| `direct_logit_attribution()` | Measure direct effect | Logit effects per square |
| `patching_metric()` | Evaluate causal interventions | Normalized score (0-1) |

### Common Patterns

**Reading what model represents at layer L**:
```python
residual = cache["resid_post", L][game, move]
board_score = residual @ probe  # (8, 8)
```

**Intervening to flip square (r, c)**:
```python
flip_dir = my_probe[:, r, c]
residual[0, move] -= 2 * (residual[0, move] @ flip_dir_normed) * flip_dir_normed
```

**Understanding neuron N at layer L**:
```python
w_in = calculate_neuron_input_weights(model, probe, L, N)  # What detects
w_out = calculate_neuron_output_weights(model, probe, L, N)  # What affects
```

### Typical Findings

| Analysis | Typical Finding |
|----------|-----------------|
| Probe accuracy vs layer | ~50% at layer 0, >90% at layer 6 |
| MLP vs Attention contributions | MLP: local features; Attn: global tracking |
| Top neurons (by std) | Detect specific 2-3 square patterns |
| Activation patching | MLP0, MLP5, MLP6, Attn7 most important |
| Direct logit attribution | Top neurons directly boost single square logits |

---

## References

- **OthelloGPT Paper**: [Emergent World Representations](https://arxiv.org/pdf/2210.13382)
- **Probing Paper**: [Learning Representations by Predicting Board States](https://arxiv.org/pdf/1610.01644)
- **Activation Patching**: [ROME (causal tracing)](https://rome.baulab.info/)
- **Original Material**: [ARENA Chapter 1 - OthelloGPT](https://arena-chapter1-transformer-interp.streamlit.app/33_🔬_[1.5.3]_OthelloGPT)
