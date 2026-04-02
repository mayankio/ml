#include "ml/optimization/SVM.hpp"

#include <fstream>

namespace ml {

LinearSVM::LinearSVM(double learning_rate, std::size_t epochs, double c, bool hard_margin)
    : learning_rate_(learning_rate), epochs_(epochs), c_(c), hard_margin_(hard_margin) {}

void LinearSVM::fit(const Matrix& features, const Matrix& targets) {
    weights_.assign(features.cols(), 0.0);
    bias_ = 0.0;
    const double c = hard_margin_ ? 1000.0 : c_;
    for (std::size_t epoch = 0; epoch < epochs_; ++epoch) {
        for (std::size_t i = 0; i < features.rows(); ++i) {
            const auto x = features.row_vector(i);
            const double y = targets(i, 0) > 0.0 ? 1.0 : -1.0;
            const double margin = y * (dot(weights_, x) + bias_);
            if (margin >= 1.0) {
                for (double& weight : weights_) {
                    weight -= learning_rate_ * weight;
                }
            } else {
                for (std::size_t j = 0; j < weights_.size(); ++j) {
                    weights_[j] -= learning_rate_ * (weights_[j] - c * y * x[j]);
                }
                bias_ += learning_rate_ * c * y;
            }
        }
    }
}

Matrix LinearSVM::decision_function(const Matrix& features) const {
    Matrix output(features.rows(), 1);
    for (std::size_t i = 0; i < features.rows(); ++i) {
        output(i, 0) = dot(weights_, features.row_vector(i)) + bias_;
    }
    return output;
}

Matrix LinearSVM::predict(const Matrix& features) const {
    Matrix scores = decision_function(features);
    for (std::size_t i = 0; i < scores.rows(); ++i) {
        scores(i, 0) = scores(i, 0) >= 0.0 ? 1.0 : 0.0;
    }
    return scores;
}

void LinearSVM::save(const std::string& path) const {
    std::ofstream out(path);
    out << learning_rate_ << ' ' << epochs_ << ' ' << c_ << ' ' << hard_margin_ << ' ' << bias_ << ' ' << weights_.size() << '\n';
    for (double weight : weights_) {
        out << weight << ' ';
    }
}

void LinearSVM::load(const std::string& path) {
    std::ifstream in(path);
    std::size_t size = 0;
    in >> learning_rate_ >> epochs_ >> c_ >> hard_margin_ >> bias_ >> size;
    weights_.assign(size, 0.0);
    for (double& weight : weights_) {
        in >> weight;
    }
}

}  // namespace ml
