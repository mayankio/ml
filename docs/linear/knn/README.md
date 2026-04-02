# K-Nearest Neighbors

## Mathematical Intuition

KNN is a non-parametric method. It stores the training set and, for a query point, finds the `k` closest samples under a distance function such as Euclidean distance:

\[
d(x, x') = \sqrt{\sum_j (x_j - x'_j)^2}
\]

Classification is done by majority vote among the nearest neighbors.

## Algorithm Steps

```text
store all training examples
for each query point
  compute distance to every training sample
  keep the k smallest distances
  vote using the neighbors' labels
return the majority label
```

## Complexity Analysis

- Training: `O(1)` after storing data
- Inference: `O(n_train * n_features)` per query
- Space: `O(n_train * n_features)`

## Usage Example

```cpp
ml::KNNClassifier model(3);
model.fit(x, y);
ml::Matrix labels = model.predict(query_points);
```
