#include "ml/linear/LogisticRegression.hpp"

#include <fstream>
#include <sstream>

#include "ml/core/ClassificationUtils.hpp"

namespace ml {

LogisticRegression::LogisticRegression(double learning_rate, std::size_t epochs)
    : learning_rate_(learning_rate), epochs_(epochs) {}

void LogisticRegression::fit(const Matrix& features, const Matrix& targets) {
    const ClassificationTargetInfo target_info = parse_classification_targets(features, targets);
    classes_ = target_info.classes;
    weights_ = Matrix(features.cols(), output_dimension_for_classes(classes_), 0.0);
    bias_.assign(weights_.cols(), 0.0);
    const double n = static_cast<double>(features.rows());

    if (is_binary_classes(classes_)) {
        for (std::size_t epoch = 0; epoch < epochs_; ++epoch) {
            std::vector<double> grad_w(features.cols(), 0.0);
            double grad_b = 0.0;
            for (std::size_t i = 0; i < features.rows(); ++i) {
                const auto x = features.row_vector(i);
                double logit = bias_[0];
                for (std::size_t j = 0; j < features.cols(); ++j) {
                    logit += weights_(j, 0) * x[j];
                }
                const double prediction = sigmoid(logit);
                const double expected = target_info.indices[i] == 1 ? 1.0 : 0.0;
                const double error = prediction - expected;
                for (std::size_t j = 0; j < features.cols(); ++j) {
                    grad_w[j] += error * x[j];
                }
                grad_b += error;
            }
            for (std::size_t j = 0; j < features.cols(); ++j) {
                weights_(j, 0) -= learning_rate_ * grad_w[j] / n;
            }
            bias_[0] -= learning_rate_ * grad_b / n;
        }
        return;
    }

    for (std::size_t epoch = 0; epoch < epochs_; ++epoch) {
        Matrix grad_w(features.cols(), classes_.size(), 0.0);
        std::vector<double> grad_b(classes_.size(), 0.0);
        for (std::size_t i = 0; i < features.rows(); ++i) {
            const auto x = features.row_vector(i);
            std::vector<double> logits(classes_.size(), 0.0);
            for (std::size_t cls = 0; cls < classes_.size(); ++cls) {
                logits[cls] = bias_[cls];
                for (std::size_t j = 0; j < features.cols(); ++j) {
                    logits[cls] += weights_(j, cls) * x[j];
                }
            }
            const std::vector<double> probabilities = softmax(logits);
            for (std::size_t cls = 0; cls < classes_.size(); ++cls) {
                const double target_value = target_info.indices[i] == cls ? 1.0 : 0.0;
                const double error = probabilities[cls] - target_value;
                for (std::size_t j = 0; j < features.cols(); ++j) {
                    grad_w(j, cls) += error * x[j];
                }
                grad_b[cls] += error;
            }
        }
        for (std::size_t j = 0; j < features.cols(); ++j) {
            for (std::size_t cls = 0; cls < classes_.size(); ++cls) {
                weights_(j, cls) -= learning_rate_ * grad_w(j, cls) / n;
            }
        }
        for (std::size_t cls = 0; cls < classes_.size(); ++cls) {
            bias_[cls] -= learning_rate_ * grad_b[cls] / n;
        }
    }
}

Matrix LogisticRegression::predict_proba(const Matrix& features) const {
    if (classes_.empty()) {
        throw std::logic_error("LogisticRegression must be fit before predict_proba");
    }

    Matrix output(features.rows(), output_dimension_for_classes(classes_));
    for (std::size_t i = 0; i < features.rows(); ++i) {
        const auto x = features.row_vector(i);
        if (is_binary_classes(classes_)) {
            double logit = bias_[0];
            for (std::size_t j = 0; j < features.cols(); ++j) {
                logit += weights_(j, 0) * x[j];
            }
            output(i, 0) = sigmoid(logit);
            continue;
        }

        std::vector<double> logits(classes_.size(), 0.0);
        for (std::size_t cls = 0; cls < classes_.size(); ++cls) {
            logits[cls] = bias_[cls];
            for (std::size_t j = 0; j < features.cols(); ++j) {
                logits[cls] += weights_(j, cls) * x[j];
            }
        }
        const std::vector<double> probabilities = softmax(logits);
        for (std::size_t cls = 0; cls < classes_.size(); ++cls) {
            output(i, cls) = probabilities[cls];
        }
    }
    return output;
}

Matrix LogisticRegression::predict(const Matrix& features) const {
    const Matrix probabilities = predict_proba(features);
    if (is_binary_classes(classes_)) {
        return decode_binary_predictions(classes_, probabilities, 0.5);
    }
    return decode_multiclass_predictions(classes_, probabilities);
}

const std::vector<int>& LogisticRegression::classes() const {
    return classes_;
}

std::size_t LogisticRegression::num_classes() const {
    return classes_.size();
}

void LogisticRegression::save(const std::string& path) const {
    std::ofstream out(path);
    out << "v2\n";
    out << learning_rate_ << ' ' << epochs_ << '\n';
    save_vector(out, classes_);
    save_vector(out, bias_);
    save_matrix(out, weights_);
}

void LogisticRegression::load(const std::string& path) {
    std::ifstream in(path);
    std::string version;
    in >> version;
    if (version == "v2") {
        in >> learning_rate_ >> epochs_;
        classes_ = load_vector<int>(in);
        bias_ = load_vector<double>(in);
        weights_ = load_matrix(in);
        return;
    }

    std::istringstream header(version);
    header >> learning_rate_;
    std::size_t size = 0;
    double legacy_bias = 0.0;
    in >> epochs_ >> legacy_bias >> size;
    weights_ = Matrix(size, 1, 0.0);
    for (std::size_t j = 0; j < size; ++j) {
        in >> weights_(j, 0);
    }
    bias_.assign(1, legacy_bias);
    classes_ = {0, 1};
}

}  // namespace ml
