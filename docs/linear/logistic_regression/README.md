# Logistic Regression

## Mathematical Intuition

Binary classification uses a linear score followed by a sigmoid:

\[
z = w^T x + b,\qquad \sigma(z)=\frac{1}{1+e^{-z}}
\]

The predicted probability is `p = sigma(z)`. The cross-entropy loss is:

\[
J(w,b)= -\frac{1}{m}\sum_{i=1}^{m}\left[y^{(i)}\log p^{(i)} + (1-y^{(i)})\log(1-p^{(i)})\right]
\]

Its gradient has the compact form:

\[
\frac{\partial J}{\partial w}=\frac{1}{m}X^T(p-y), \qquad
\frac{\partial J}{\partial b}=\frac{1}{m}\sum (p-y)
\]

## Algorithm Steps

```text
initialize weights and bias
repeat for each epoch
  compute sigmoid probabilities
  measure classification error p - y
  update weights and bias
threshold probabilities at 0.5 for class labels
```

## Complexity Analysis

- Training: `O(epochs * n_samples * n_features)`
- Inference: `O(n_samples * n_features)`
- Space: `O(n_features)`

## Usage Example

```cpp
ml::LogisticRegression model(0.1, 2000);
model.fit(x, y);
ml::Matrix classes = model.predict(x);
```
