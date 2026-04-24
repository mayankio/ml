#include "ml/optimization/SVM.hpp"

#include <fstream>
#include <sstream>

#include "ml/core/ClassificationUtils.hpp"

namespace ml {

LinearSVM::LinearSVM(double learning_rate, std::size_t epochs, double c, bool hard_margin)
    : learning_rate_(learning_rate), epochs_(epochs), c_(c), hard_margin_(hard_margin) {}

void LinearSVM::fit(const Matrix& features, const Matrix& targets) {
    const ClassificationTargetInfo target_info = parse_classification_targets(features, targets);
    classes_ = target_info.classes;
    weights_ = Matrix(features.cols(), output_dimension_for_classes(classes_), 0.0);
    bias_.assign(weights_.cols(), 0.0);
    const double c = hard_margin_ ? 1000.0 : c_;

    if (is_binary_classes(classes_)) {
        for (std::size_t epoch = 0; epoch < epochs_; ++epoch) {
            for (std::size_t i = 0; i < features.rows(); ++i) {
                const auto x = features.row_vector(i);
                const double y = target_info.indices[i] == 1 ? 1.0 : -1.0;
                double score = bias_[0];
                for (std::size_t j = 0; j < features.cols(); ++j) {
                    score += weights_(j, 0) * x[j];
                }
                const double margin = y * score;
                if (margin >= 1.0) {
                    for (std::size_t j = 0; j < weights_.rows(); ++j) {
                        weights_(j, 0) -= learning_rate_ * weights_(j, 0);
                    }
                } else {
                    for (std::size_t j = 0; j < weights_.rows(); ++j) {
                        weights_(j, 0) -= learning_rate_ * (weights_(j, 0) - c * y * x[j]);
                    }
                    bias_[0] += learning_rate_ * c * y;
                }
            }
        }
        return;
    }

    for (std::size_t epoch = 0; epoch < epochs_; ++epoch) {
        for (std::size_t i = 0; i < features.rows(); ++i) {
            const auto x = features.row_vector(i);
            for (std::size_t cls = 0; cls < classes_.size(); ++cls) {
                const double y = target_info.indices[i] == cls ? 1.0 : -1.0;
                double score = bias_[cls];
                for (std::size_t j = 0; j < features.cols(); ++j) {
                    score += weights_(j, cls) * x[j];
                }
                const double margin = y * score;
                if (margin >= 1.0) {
                    for (std::size_t j = 0; j < weights_.rows(); ++j) {
                        weights_(j, cls) -= learning_rate_ * weights_(j, cls);
                    }
                } else {
                    for (std::size_t j = 0; j < features.cols(); ++j) {
                        weights_(j, cls) -= learning_rate_ * (weights_(j, cls) - c * y * x[j]);
                    }
                    bias_[cls] += learning_rate_ * c * y;
                }
            }
        }
    }
}

Matrix LinearSVM::decision_function(const Matrix& features) const {
    if (classes_.empty()) {
        throw std::logic_error("LinearSVM must be fit before decision_function");
    }
    Matrix output(features.rows(), output_dimension_for_classes(classes_));
    for (std::size_t i = 0; i < features.rows(); ++i) {
        const auto x = features.row_vector(i);
        if (is_binary_classes(classes_)) {
            double score = bias_[0];
            for (std::size_t j = 0; j < features.cols(); ++j) {
                score += weights_(j, 0) * x[j];
            }
            output(i, 0) = score;
            continue;
        }

        for (std::size_t cls = 0; cls < classes_.size(); ++cls) {
            output(i, cls) = bias_[cls];
            for (std::size_t j = 0; j < features.cols(); ++j) {
                output(i, cls) += weights_(j, cls) * x[j];
            }
        }
    }
    return output;
}

Matrix LinearSVM::predict(const Matrix& features) const {
    const Matrix scores = decision_function(features);
    if (is_binary_classes(classes_)) {
        return decode_binary_predictions(classes_, scores, 0.0);
    }
    return decode_multiclass_predictions(classes_, scores);
}

const std::vector<int>& LinearSVM::classes() const {
    return classes_;
}

std::size_t LinearSVM::num_classes() const {
    return classes_.size();
}

void LinearSVM::save(const std::string& path) const {
    std::ofstream out(path);
    out << "v2\n";
    out << learning_rate_ << ' ' << epochs_ << ' ' << c_ << ' ' << hard_margin_ << '\n';
    save_vector(out, classes_);
    save_vector(out, bias_);
    save_matrix(out, weights_);
}

void LinearSVM::load(const std::string& path) {
    std::ifstream in(path);
    std::string version;
    in >> version;
    if (version == "v2") {
        in >> learning_rate_ >> epochs_ >> c_ >> hard_margin_;
        classes_ = load_vector<int>(in);
        bias_ = load_vector<double>(in);
        weights_ = load_matrix(in);
        return;
    }

    std::istringstream header(version);
    header >> learning_rate_;
    std::size_t size = 0;
    double legacy_bias = 0.0;
    in >> epochs_ >> c_ >> hard_margin_ >> legacy_bias >> size;
    weights_ = Matrix(size, 1, 0.0);
    for (std::size_t j = 0; j < size; ++j) {
        in >> weights_(j, 0);
    }
    bias_.assign(1, legacy_bias);
    classes_ = {0, 1};
}

}  // namespace ml
