# Random Forest

## Mathematical Intuition

Random forests reduce variance by averaging many decision trees trained on slightly different views of the data:

- Bootstrap sampling: each tree sees a sampled version of the training set.
- Feature subsampling: each split only considers a random subset of features.

For classification, the final prediction is a majority vote over trees.

## Algorithm Steps

```text
for each tree
  draw a bootstrap sample from the training set
  train a decision tree on that sample
  limit split candidates to a random feature subset
for prediction
  ask every tree for a class label
  return the majority vote
```

## Complexity Analysis

- Training: about `O(n_trees * tree_training_cost)`
- Inference: `O(n_trees * tree_depth)` per sample
- Space: `O(total_nodes_across_all_trees)`

## Usage Example

```cpp
ml::RandomForestClassifier forest(25, 6, 2);
forest.fit(x, y);
ml::Matrix labels = forest.predict(x_test);
```
