# Gradient Boosting Machines

## Mathematical Intuition

Gradient boosting builds an additive model:

\[
F_M(x)=F_0(x)+\eta\sum_{m=1}^{M} h_m(x)
\]

For squared error regression, the negative gradient is just the residual:

\[
r_i = y_i - F(x_i)
\]

Each new weak learner is trained to predict the current residuals.

## Algorithm Steps

```text
initialize prediction with the mean target value
repeat for each boosting round
  compute residuals y - current_prediction
  fit a shallow regression tree to residuals
  add a scaled version of that tree's output
sum all trees during inference
```

## Complexity Analysis

- Training: `O(n_estimators * weak_learner_training_cost)`
- Inference: `O(n_estimators * tree_depth)` per sample
- Space: `O(total_nodes_across_trees)`

## Usage Example

```cpp
ml::GradientBoostingRegressor gbm(50, 0.1, 2);
gbm.fit(x, y);
ml::Matrix preds = gbm.predict(x_test);
```
