# Gaussian Naive Bayes

## Mathematical Intuition

Naive Bayes applies Bayes' rule:

\[
P(y \mid x) \propto P(y)\prod_j P(x_j \mid y)
\]

The "naive" assumption is conditional independence of features given the class. For continuous features, we often use a Gaussian likelihood:

\[
P(x_j \mid y=c)=\frac{1}{\sqrt{2\pi\sigma_{cj}^2}}
\exp\left(-\frac{(x_j-\mu_{cj})^2}{2\sigma_{cj}^2}\right)
\]

The model estimates class priors, means, and variances from the data.

## Algorithm Steps

```text
group training samples by class
estimate prior probability for each class
estimate Gaussian mean and variance for every feature in every class
for a new sample
  compute log posterior for each class
  choose the class with the largest score
```

## Complexity Analysis

- Training: `O(n_samples * n_features)`
- Inference: `O(n_classes * n_features)` per sample
- Space: `O(n_classes * n_features)`

## Usage Example

```cpp
ml::GaussianNaiveBayes model;
model.fit(x, y);
ml::Matrix labels = model.predict(x_test);
```
