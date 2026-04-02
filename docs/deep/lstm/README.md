# Simple LSTM

## Mathematical Intuition

An LSTM adds gates that control memory flow:

\[
f_t = \sigma(W_f [x_t, h_{t-1}] + b_f)
\]

\[
i_t = \sigma(W_i [x_t, h_{t-1}] + b_i), \qquad
\tilde{c}_t = \tanh(W_c [x_t, h_{t-1}] + b_c)
\]

\[
c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t
\]

\[
o_t = \sigma(W_o [x_t, h_{t-1}] + b_o), \qquad
h_t = o_t \odot \tanh(c_t)
\]

This gating helps LSTMs preserve information over longer sequences than a plain RNN.

## Algorithm Steps

```text
start with zero hidden state and cell state
for each time step
  compute forget, input, output, and candidate gates
  update memory cell
  update hidden state
map final hidden state to an output probability
```

## Complexity Analysis

- Training: `O(epochs * n_samples * sequence_length * hidden_dim * (input_dim + hidden_dim))`
- Inference: same forward-pass order
- Space: `O(hidden_dim * (input_dim + hidden_dim))`

## Usage Example

```cpp
ml::SimpleLSTM lstm(10, 4, 16, 0.03, 1000);
lstm.fit(sequence_matrix, labels);
ml::Matrix probs = lstm.predict_proba(sequence_matrix);
```
