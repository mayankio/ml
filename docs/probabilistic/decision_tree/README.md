# Decision Tree

## Mathematical Intuition

A decision tree recursively splits the feature space to reduce impurity. For classification, this implementation uses Gini impurity:

\[
G(S)=1-\sum_k p_k^2
\]

At each node, we search for the feature and threshold that produce the lowest weighted child impurity.

For regression trees, the split quality is based on variance reduction.

## Algorithm Steps

```text
start with all samples at the root
for each candidate feature and threshold
  split samples into left and right groups
  compute weighted impurity after the split
pick the best split
recurse until depth or sample stopping rule is hit
store a class label or mean value in each leaf
```

## Complexity Analysis

- Training: roughly `O(n_samples * n_features * candidate_splits * depth)`
- Inference: `O(depth)` per sample
- Space: `O(number_of_nodes)`

## Usage Example

```cpp
ml::DecisionTreeClassifier tree(5, 2);
tree.fit(x, y);
ml::Matrix labels = tree.predict(x_test);
```
