# K-Means Clustering

## Mathematical Intuition

K-means partitions data into `k` clusters by minimizing within-cluster squared distance:

\[
\sum_{i=1}^{n}\|x^{(i)}-\mu_{c(i)}\|^2
\]

Each cluster is represented by its centroid `mu`.

## Algorithm Steps

```text
initialize k centroids
repeat
  assign each sample to its closest centroid
  recompute each centroid as the mean of its assigned points
stop after convergence or max iterations
```

## Complexity Analysis

- Training: `O(max_iters * n_samples * n_clusters * n_features)`
- Inference: `O(n_samples * n_clusters * n_features)`
- Space: `O(n_clusters * n_features)`

## Usage Example

```cpp
ml::KMeans model(3, 100);
model.fit(x, ml::Matrix{});
ml::Matrix clusters = model.predict(x);
```
