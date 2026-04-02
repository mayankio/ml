# Self-Attention

## Mathematical Intuition

Self-attention lets each token look at every other token. Starting from input embeddings `X`, we compute:

\[
Q = XW_Q,\qquad K = XW_K,\qquad V = XW_V
\]

Attention weights come from scaled dot products:

\[
A = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)
\]

The output is:

\[
\text{Attention}(X) = A V W_O
\]

This mechanism is powerful because it can connect distant tokens in a single step.

## Algorithm Steps

```text
project input embeddings into queries, keys, and values
compute pairwise attention scores
apply softmax row-wise to get attention weights
mix value vectors using those weights
optionally project back to the embedding size
```

## Complexity Analysis

- Forward pass: `O(sequence_length^2 * projection_dim)`
- Space: `O(sequence_length^2)` for the attention matrix

## Usage Example

```cpp
ml::SelfAttention attention(seq_len, embed_dim, proj_dim);
ml::Matrix contextual = attention.predict(sequence_embeddings);
```
