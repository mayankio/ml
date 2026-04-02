#include "ml/deep/CNN.hpp"

#include <fstream>

namespace ml {

SimpleCNN::SimpleCNN(std::size_t image_height, std::size_t image_width, std::size_t num_filters, std::size_t kernel_size, double learning_rate, std::size_t epochs)
    : image_height_(image_height),
      image_width_(image_width),
      num_filters_(num_filters),
      kernel_size_(kernel_size),
      learning_rate_(learning_rate),
      epochs_(epochs),
      dense_weights_(num_filters_, 1, 0.0) {
    for (std::size_t filter = 0; filter < num_filters_; ++filter) {
        filters_.push_back(Matrix::random(kernel_size_, kernel_size_, -0.3, 0.3, static_cast<std::uint32_t>(filter + 10)));
    }
}

std::vector<double> SimpleCNN::forward_single(const std::vector<double>& input, std::vector<double>* hidden) const {
    const std::size_t conv_h = image_height_ - kernel_size_ + 1;
    const std::size_t conv_w = image_width_ - kernel_size_ + 1;
    std::vector<double> pooled(num_filters_, 0.0);
    for (std::size_t filter = 0; filter < num_filters_; ++filter) {
        double best = -1e18;
        for (std::size_t i = 0; i < conv_h; ++i) {
            for (std::size_t j = 0; j < conv_w; ++j) {
                double sum = 0.0;
                for (std::size_t ki = 0; ki < kernel_size_; ++ki) {
                    for (std::size_t kj = 0; kj < kernel_size_; ++kj) {
                        const std::size_t input_index = (i + ki) * image_width_ + (j + kj);
                        sum += input[input_index] * filters_[filter](ki, kj);
                    }
                }
                best = std::max(best, relu(sum));
            }
        }
        pooled[filter] = best;
    }
    if (hidden != nullptr) {
        *hidden = pooled;
    }
    return pooled;
}

void SimpleCNN::fit(const Matrix& features, const Matrix& targets) {
    for (std::size_t epoch = 0; epoch < epochs_; ++epoch) {
        for (std::size_t i = 0; i < features.rows(); ++i) {
            std::vector<double> pooled;
            const auto input = features.row_vector(i);
            forward_single(input, &pooled);
            double logit = dense_bias_;
            for (std::size_t f = 0; f < num_filters_; ++f) {
                logit += pooled[f] * dense_weights_(f, 0);
            }
            const double prediction = sigmoid(logit);
            const double error = prediction - targets(i, 0);
            for (std::size_t f = 0; f < num_filters_; ++f) {
                dense_weights_(f, 0) -= learning_rate_ * error * pooled[f];
            }
            dense_bias_ -= learning_rate_ * error;
        }
    }
}

Matrix SimpleCNN::predict_proba(const Matrix& features) const {
    Matrix output(features.rows(), 1);
    for (std::size_t i = 0; i < features.rows(); ++i) {
        const auto pooled = forward_single(features.row_vector(i));
        double logit = dense_bias_;
        for (std::size_t f = 0; f < num_filters_; ++f) {
            logit += pooled[f] * dense_weights_(f, 0);
        }
        output(i, 0) = sigmoid(logit);
    }
    return output;
}

Matrix SimpleCNN::predict(const Matrix& features) const {
    Matrix probabilities = predict_proba(features);
    for (std::size_t i = 0; i < probabilities.rows(); ++i) {
        probabilities(i, 0) = probabilities(i, 0) >= 0.5 ? 1.0 : 0.0;
    }
    return probabilities;
}

void SimpleCNN::save(const std::string& path) const {
    std::ofstream out(path);
    out << image_height_ << ' ' << image_width_ << ' ' << num_filters_ << ' ' << kernel_size_ << ' ' << learning_rate_ << ' ' << epochs_ << ' ' << dense_bias_ << '\n';
    for (const auto& filter : filters_) {
        save_matrix(out, filter);
    }
    save_matrix(out, dense_weights_);
}

void SimpleCNN::load(const std::string& path) {
    std::ifstream in(path);
    in >> image_height_ >> image_width_ >> num_filters_ >> kernel_size_ >> learning_rate_ >> epochs_ >> dense_bias_;
    filters_.clear();
    for (std::size_t filter = 0; filter < num_filters_; ++filter) {
        filters_.push_back(load_matrix(in));
    }
    dense_weights_ = load_matrix(in);
}

}  // namespace ml
