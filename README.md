# Modular Transformer Implementation

This repository contains a modular, from-scratch implementation of Transformer models in PyTorch. It features the three primary architectures—**Seq2Seq** (Encoder-Decoder), **BERT** (Encoder-only), and **GPT** (Decoder-only)—built from shared, reusable components.

The implementation includes advanced optimizations like **Weight Tying** and **KV-Caching** for fast autoregressive generation, as well as robust checkpointing utilities.

## 🏗️ Architecture Overview

The system is built on a hierarchy of classes, starting from atomic layers up to full model architectures.

### 1. Core Components
These base classes are used across all model types.

* **`PositionalEncoding`**: Injects sequence order information into the embeddings.
    * Uses sinusoidal functions so the model can learn to attend by relative positions.
    * $PE_{(pos, 2i)} = \sin(pos / 10000^{2i/d_{model}})$
    * $PE_{(pos, 2i+1)} = \cos(pos / 10000^{2i/d_{model}})$

* **`MultiHeadAttention` (MHA)**: The core mechanism allowing the model to focus on different parts of the input sequence.
    * Supports **Self-Attention** (query=key=value) and **Cross-Attention** (query from decoder, key/value from encoder).
    * **KV-Caching**: During inference, past Keys and Values are stored to avoid re-computation.
    * **Math**:
        $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

* **`PositionwiseFeedForward`**: A two-layer fully connected network applied to each position independently.
    * Structure: Linear $\rightarrow$ ReLU $\rightarrow$ Linear.
    * **Math**:
        $$\text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2$$

### 2. Layers & Stacks
* **`EncoderLayer`**: Stacks MHA + FeedForward with Residual Connections and Layer Normalization.
    * `LayerNorm(x + Sublayer(x))`
* **`DecoderLayer`**: Similar to EncoderLayer but adds a third sub-layer for **Cross-Attention** (attending to encoder outputs). It also supports masking for causal (autoregressive) generation.
* **`Encoder` / `Decoder`**: Containers that stack $N$ layers and handle embedding + positional encoding.

### 3. Models
* **`Seq2SeqTransformer`**: Standard Encoder-Decoder architecture (e.g., for Translation).
* **`BERTModel`**: Encoder stack only. Used for understanding tasks (Masked Language Modeling).
* **`GPTModel`**: Decoder stack only. Used for generation tasks (Causal Language Modeling).

---

## 🚀 Key Features

### Weight Tying
To reduce parameter count and improve regularization, we tie the weights of the embedding layer and the final output projection layer (Softmax layer).
$$W_{embedding} = W_{prediction\_head}$$
*Implemented in all three model classes.*

### KV-Caching (Optimization)
In autoregressive generation (GPT), generating token $t_{100}$ normally requires processing tokens $t_1...t_{99}$ again. KV-Caching saves the Key and Value matrices of past tokens.
* **Without Cache**: Complexity $O(N^2)$
* **With Cache**: Complexity $O(N)$ per step (linear time generation).

---

## 📝 Usage Guide

### 1. Training BERT (Masked Language Modeling)
BERT is trained to predict tokens that have been randomly replaced with `[MASK]`.

```python
# Create mask for 15% of tokens
prob_matrix = torch.full(inputs.shape, 0.15)
masked_indices = torch.bernoulli(prob_matrix).bool()
inputs[masked_indices] = tokenizer.token_to_id("[MASK]")

# Forward pass
logits = bert_model(inputs, padding_mask)

# Calculate Loss (only on masked tokens)
loss = criterion(logits, labels)

```

### 2. Training GPT (Next Token Prediction)

GPT is trained to predict the next token in the sequence. The targets are simply the inputs shifted by one position.

```python
# Shift inputs and targets
inputs = batch[:, :-1]   # t_0 ... t_N-1
targets = batch[:, 1:]   # t_1 ... t_N

# Forward pass (Causal masking is handled internally)
logits, _ = gpt_model(inputs)

# Calculate Loss
loss = criterion(logits, targets)

```

### 3. Inference with KV-Caching

To generate text efficiently, use the provided generation loop.

```python
from transformers_script import generate_text_kv_cache

prompt = "The robot walked"
output = generate_text_kv_cache(gpt_model, prompt, max_tokens=20)
print(output)

```

---

## 💾 Saving & Loading Checkpoints

We provide utility functions to save the model state, optimizer state, and training metadata.

### Saving

```python
from transformers_script import save_checkpoint

# Saves model state, optimizer state, epoch, and loss
save_checkpoint(model, optimizer, epoch=5, loss=0.45, filepath="checkpoint.pth")

```

### Loading for Training (Resuming)

```python
from transformers_script import load_checkpoint

start_epoch, last_loss = load_checkpoint(model, optimizer, "checkpoint.pth")
# Resume training loop from `start_epoch`

```

### Loading for Inference

```python
# Pass optimizer=None to only load model weights
load_checkpoint(model, optimizer=None, filepath="checkpoint.pth")
model.eval() # Important!

```

---

## 📐 Mathematical Reference

### Scaled Dot-Product Attention

The attention scores are calculated as:

Where  is the dimension of the key vector. Dividing by  prevents the dot products from growing too large in magnitude, which would push the Softmax function into regions with extremely small gradients.

### Layer Normalization

Applied after each sub-layer:

Where  and  are the mean and variance of the input , and  are learnable parameters.

---

## 📦 Requirements

* `torch`
* `tokenizers`
