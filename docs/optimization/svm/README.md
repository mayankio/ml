# Linear Support Vector Machine

## Mathematical Intuition

For binary classes `y in {-1, +1}`, a linear SVM looks for a separating hyperplane:

\[
f(x)=w^T x + b
\]

The soft-margin objective combines margin maximization with hinge loss:

\[
\min_{w,b}\ \frac{1}{2}\|w\|^2 + C\sum_i \max(0, 1-y^{(i)}f(x^{(i)}))
\]

Hard margin is the special case where violations are effectively not allowed, which only works on perfectly separable data.

For multiclass datasets this implementation trains one linear SVM per class in a one-vs-rest setup and predicts the class whose score is largest.

## Algorithm Steps

```text
initialize w and b
for each sample, epoch, and class head
  compute margin y * (w^T x + b)
  if margin >= 1, shrink weights toward smaller norm
  otherwise update using hinge-loss gradient
predict using the sign of the score for binary
or argmax over class scores for multiclass
```

## Complexity Analysis

- Training: `O(epochs * n_samples * n_features)`
- Inference: `O(n_samples * n_features)`
- Space: `O(n_features * n_classes)`

## Usage Example

```cpp
ml::LinearSVM svm(0.01, 2000, 1.0, false);
svm.fit(x, y);
ml::Matrix labels = svm.predict(x_test);
ml::Matrix scores = svm.decision_function(x_test);
```
