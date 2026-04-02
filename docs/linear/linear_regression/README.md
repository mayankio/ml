# Linear Regression

## Mathematical Intuition

We model the prediction as:

\[
\hat{y} = w^T x + b
\]

The mean squared error loss is:

\[
J(w, b) = \frac{1}{m}\sum_{i=1}^{m}(\hat{y}^{(i)} - y^{(i)})^2
\]

Gradient descent updates:

\[
w_j \leftarrow w_j - \alpha \frac{2}{m}\sum_{i=1}^{m}(\hat{y}^{(i)} - y^{(i)})x_j^{(i)}
\]

\[
b \leftarrow b - \alpha \frac{2}{m}\sum_{i=1}^{m}(\hat{y}^{(i)} - y^{(i)})
\]

## Algorithm Steps

```text
initialize w and b
repeat for each epoch
  compute predictions on all samples
  compute gradients of MSE with respect to w and b
  update parameters with gradient descent
return learned line
```

## Complexity Analysis

- Training: `O(epochs * n_samples * n_features)`
- Inference: `O(n_samples * n_features)`
- Space: `O(n_features)`

## Usage Example

```cpp
ml::LinearRegression model(0.01, 3000);
model.fit(x, y);
ml::Matrix preds = model.predict(x);
```
