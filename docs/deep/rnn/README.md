# Simple RNN

## Mathematical Intuition

A recurrent neural network updates a hidden state over time:

\[
h_t = \tanh(W_x x_t + W_h h_{t-1} + b_h)
\]

After the final time step, a dense output layer maps the hidden state to a prediction:

\[
\hat{y} = \sigma(W_y h_T + b_y)
\]

The hidden state acts like a running summary of the sequence.

## Algorithm Steps

```text
initialize hidden state to zeros
for each time step
  mix current input with previous hidden state
  apply tanh to get new hidden state
use final hidden state for prediction
update output layer during training
```

## Complexity Analysis

- Training: `O(epochs * n_samples * sequence_length * hidden_dim * input_dim)`
- Inference: same order without parameter updates
- Space: `O(hidden_dim^2 + input_dim * hidden_dim)`

## Usage Example

```cpp
ml::SimpleRNN rnn(10, 4, 16, 0.05, 800);
rnn.fit(sequence_matrix, labels);
ml::Matrix probs = rnn.predict_proba(sequence_matrix);
```
