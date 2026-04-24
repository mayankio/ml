#include "ml/deep/CNN.hpp"

#include <fstream>
#include <sstream>

#include "ml/core/ClassificationUtils.hpp"

namespace ml {

SimpleCNN::SimpleCNN(std::size_t image_height, std::size_t image_width, std::size_t num_filters, std::size_t kernel_size, double learning_rate, std::size_t epochs)
    : image_height_(image_height),
      image_width_(image_width),
      num_filters_(num_filters),
      kernel_size_(kernel_size),
      learning_rate_(learning_rate),
      epochs_(epochs),
      dense_weights_(num_filters_, 1, 0.0),
      dense_bias_(1, 0.0) {
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
    const ClassificationTargetInfo target_info = parse_classification_targets(features, targets);
    classes_ = target_info.classes;
    const std::size_t output_dim = output_dimension_for_classes(classes_);
    if (dense_weights_.rows() != num_filters_ || dense_weights_.cols() != output_dim) {
        dense_weights_ = Matrix(num_filters_, output_dim, 0.0);
    }
    if (dense_bias_.size() != output_dim) {
        dense_bias_.assign(output_dim, 0.0);
    }

    for (std::size_t epoch = 0; epoch < epochs_; ++epoch) {
        for (std::size_t i = 0; i < features.rows(); ++i) {
            std::vector<double> pooled;
            const auto input = features.row_vector(i);
            forward_single(input, &pooled);

            if (is_binary_classes(classes_)) {
                double logit = dense_bias_[0];
                for (std::size_t f = 0; f < num_filters_; ++f) {
                    logit += pooled[f] * dense_weights_(f, 0);
                }
                const double prediction = sigmoid(logit);
                const double error = prediction - (target_info.indices[i] == 1 ? 1.0 : 0.0);
                for (std::size_t f = 0; f < num_filters_; ++f) {
                    dense_weights_(f, 0) -= learning_rate_ * error * pooled[f];
                }
                dense_bias_[0] -= learning_rate_ * error;
                continue;
            }

            std::vector<double> logits(classes_.size(), 0.0);
            for (std::size_t cls = 0; cls < classes_.size(); ++cls) {
                logits[cls] = dense_bias_[cls];
                for (std::size_t f = 0; f < num_filters_; ++f) {
                    logits[cls] += pooled[f] * dense_weights_(f, cls);
                }
            }
            const std::vector<double> probabilities = softmax(logits);
            for (std::size_t cls = 0; cls < classes_.size(); ++cls) {
                const double error = probabilities[cls] - (target_info.indices[i] == cls ? 1.0 : 0.0);
                for (std::size_t f = 0; f < num_filters_; ++f) {
                    dense_weights_(f, cls) -= learning_rate_ * error * pooled[f];
                }
                dense_bias_[cls] -= learning_rate_ * error;
            }
        }
    }
}

Matrix SimpleCNN::predict_proba(const Matrix& features) const {
    if (classes_.empty()) {
        throw std::logic_error("SimpleCNN must be fit before predict_proba");
    }
    Matrix output(features.rows(), output_dimension_for_classes(classes_));
    for (std::size_t i = 0; i < features.rows(); ++i) {
        const auto pooled = forward_single(features.row_vector(i));
        if (is_binary_classes(classes_)) {
            double logit = dense_bias_[0];
            for (std::size_t f = 0; f < num_filters_; ++f) {
                logit += pooled[f] * dense_weights_(f, 0);
            }
            output(i, 0) = sigmoid(logit);
            continue;
        }

        std::vector<double> logits(classes_.size(), 0.0);
        for (std::size_t cls = 0; cls < classes_.size(); ++cls) {
            logits[cls] = dense_bias_[cls];
            for (std::size_t f = 0; f < num_filters_; ++f) {
                logits[cls] += pooled[f] * dense_weights_(f, cls);
            }
        }
        const std::vector<double> probabilities = softmax(logits);
        for (std::size_t cls = 0; cls < classes_.size(); ++cls) {
            output(i, cls) = probabilities[cls];
        }
    }
    return output;
}

Matrix SimpleCNN::predict(const Matrix& features) const {
    const Matrix probabilities = predict_proba(features);
    if (is_binary_classes(classes_)) {
        return decode_binary_predictions(classes_, probabilities, 0.5);
    }
    return decode_multiclass_predictions(classes_, probabilities);
}

const std::vector<int>& SimpleCNN::classes() const {
    return classes_;
}

std::size_t SimpleCNN::num_classes() const {
    return classes_.size();
}

void SimpleCNN::save(const std::string& path) const {
    std::ofstream out(path);
    out << "v2\n";
    out << image_height_ << ' ' << image_width_ << ' ' << num_filters_ << ' ' << kernel_size_ << ' ' << learning_rate_ << ' ' << epochs_ << '\n';
    save_vector(out, classes_);
    save_vector(out, dense_bias_);
    for (const auto& filter : filters_) {
        save_matrix(out, filter);
    }
    save_matrix(out, dense_weights_);
}

void SimpleCNN::load(const std::string& path) {
    std::ifstream in(path);
    std::string version;
    in >> version;
    if (version == "v2") {
        in >> image_height_ >> image_width_ >> num_filters_ >> kernel_size_ >> learning_rate_ >> epochs_;
        classes_ = load_vector<int>(in);
        dense_bias_ = load_vector<double>(in);
        filters_.clear();
        for (std::size_t filter = 0; filter < num_filters_; ++filter) {
            filters_.push_back(load_matrix(in));
        }
        dense_weights_ = load_matrix(in);
        return;
    }

    std::istringstream header(version);
    header >> image_height_;
    double legacy_bias = 0.0;
    in >> image_width_ >> num_filters_ >> kernel_size_ >> learning_rate_ >> epochs_ >> legacy_bias;
    filters_.clear();
    for (std::size_t filter = 0; filter < num_filters_; ++filter) {
        filters_.push_back(load_matrix(in));
    }
    dense_weights_ = load_matrix(in);
    dense_bias_.assign(1, legacy_bias);
    classes_ = {0, 1};
}

}  // namespace ml
