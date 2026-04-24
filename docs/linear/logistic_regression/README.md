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

For `K > 2` classes, the model switches to a softmax output:

\[
z_k = w_k^T x + b_k,\qquad
p_k = \frac{e^{z_k}}{\sum_j e^{z_j}}
\]

Training then minimizes multiclass cross-entropy and `predict` returns the class id with the largest probability.

## Algorithm Steps

```text
initialize weights and bias
repeat for each epoch
  compute sigmoid or softmax probabilities
  measure classification error p - y
  update weights and bias
threshold probabilities at 0.5 for binary labels
or take argmax for multiclass labels
```

## Complexity Analysis

- Training: `O(epochs * n_samples * n_features)`
- Inference: `O(n_samples * n_features)`
- Space: `O(n_features * n_classes)`

## Usage Example

```cpp
ml::LogisticRegression model(0.1, 2000);
model.fit(x, y);
ml::Matrix classes = model.predict(x);
ml::Matrix probs = model.predict_proba(x);
```
