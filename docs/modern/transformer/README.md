# Transformer Block

## Mathematical Intuition

A transformer block combines self-attention with a position-wise feed-forward network. In encoder form:

\[
H = \text{SelfAttention}(X)
\]

\[
Z = \phi(HW_1 + b_1)W_2 + b_2
\]

Decoder blocks add masked attention and often cross-attention to encoder outputs. This educational implementation focuses on the self-attention plus feed-forward pattern and then pools the encoded sequence for classification.

## Algorithm Steps

```text
apply self-attention across the sequence
pass token representations through a feed-forward network
pool token states into a single sequence summary
map the summary to an output probability
```

## Complexity Analysis

- Forward pass: `O(sequence_length^2 * projection_dim + sequence_length * embedding_dim * hidden_dim)`
- Inference: same order as forward pass
- Space: `O(sequence_length^2 + embedding_dim * hidden_dim)`

## Usage Example

```cpp
ml::TransformerClassifier model(seq_len, embed_dim, proj_dim, hidden_dim);
model.fit(flattened_sequence_matrix, labels);
ml::Matrix probs = model.predict_proba(flattened_sequence_matrix);
```
