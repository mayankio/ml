# Simple CNN

## Mathematical Intuition

Convolutional neural networks reuse local filters across the image. A filter computes:

\[
s_{ij}^{(k)} = \sum_{u,v} x_{i+u,j+v} \cdot w_{uv}^{(k)}
\]

This implementation applies ReLU after convolution and then uses max pooling by taking the strongest activation from each filter map. A dense readout layer performs sigmoid binary classification or softmax multiclass classification.

## Algorithm Steps

```text
slide each filter over the image
apply ReLU to convolution responses
keep the strongest response per filter
feed pooled values into a dense output head
update the output layer during training
```

## Complexity Analysis

- Training: `O(epochs * n_samples * num_filters * conv_area * kernel_area)`
- Inference: similar forward-pass cost
- Space: `O(num_filters * kernel_size^2)`

## Usage Example

```cpp
ml::SimpleCNN cnn(3, 3, 4, 2, 0.05, 500);
cnn.fit(flattened_images, labels);
ml::Matrix probs = cnn.predict_proba(flattened_images);
ml::Matrix classes = cnn.predict(flattened_images);
```
