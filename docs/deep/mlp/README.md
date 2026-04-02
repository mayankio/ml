# Multi-Layer Perceptron

## Mathematical Intuition

An MLP stacks dense layers and nonlinear activations:

\[
z_1 = XW_1 + b_1,\qquad a_1 = \sigma(z_1)
\]

\[
z_2 = a_1 W_2 + b_2,\qquad \hat{y} = \sigma(z_2)
\]

Backpropagation uses the chain rule to move gradients from the output layer back into the hidden layer.

## Algorithm Steps

```text
initialize dense layer weights
repeat for each sample and epoch
  run forward pass through hidden and output layers
  compute output error
  backpropagate into hidden activations
  update both layers' weights and biases
```

## Complexity Analysis

- Training: `O(epochs * n_samples * (input_dim * hidden_dim + hidden_dim))`
- Inference: `O(n_samples * input_dim * hidden_dim)`
- Space: `O(input_dim * hidden_dim)`

## Usage Example

```cpp
ml::MLPClassifier mlp(2, 8, 0.1, 3000);
mlp.fit(x, y);
ml::Matrix labels = mlp.predict(x_test);
```
