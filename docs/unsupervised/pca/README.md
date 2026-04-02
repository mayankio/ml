# Principal Component Analysis

## Mathematical Intuition

PCA finds directions of maximum variance. After centering the data, we form the covariance matrix:

\[
\Sigma = \frac{1}{m-1}X^T X
\]

Principal components are eigenvectors of `Sigma`. Projecting onto the top components gives a lower-dimensional representation:

\[
Z = X W_k
\]

This implementation uses power iteration to approximate the dominant eigenvectors.

## Algorithm Steps

```text
center the data by subtracting feature means
compute covariance matrix
for each requested component
  run power iteration to estimate the dominant eigenvector
  deflate the covariance matrix
project centered data onto learned components
```

## Complexity Analysis

- Training: about `O(power_iterations * n_components * n_features^2)`
- Inference: `O(n_samples * n_features * n_components)`
- Space: `O(n_features^2)`

## Usage Example

```cpp
ml::PCA pca(2);
pca.fit(x, ml::Matrix{});
ml::Matrix reduced = pca.transform(x);
```
