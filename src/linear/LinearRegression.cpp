#include "ml/linear/LinearRegression.hpp"

#include <fstream>

namespace ml {

LinearRegression::LinearRegression(double learning_rate, std::size_t epochs)
    : learning_rate_(learning_rate), epochs_(epochs) {}

void LinearRegression::fit(const Matrix& features, const Matrix& targets) {
    weights_.assign(features.cols(), 0.0);
    bias_ = 0.0;
    const double n = static_cast<double>(features.rows());
    for (std::size_t epoch = 0; epoch < epochs_; ++epoch) {
        std::vector<double> grad_w(features.cols(), 0.0);
        double grad_b = 0.0;
        for (std::size_t i = 0; i < features.rows(); ++i) {
            const auto x = features.row_vector(i);
            const double prediction = dot(weights_, x) + bias_;
            const double error = prediction - targets(i, 0);
            for (std::size_t j = 0; j < features.cols(); ++j) {
                grad_w[j] += error * x[j];
            }
            grad_b += error;
        }
        for (std::size_t j = 0; j < weights_.size(); ++j) {
            weights_[j] -= learning_rate_ * (2.0 / n) * grad_w[j];
        }
        bias_ -= learning_rate_ * (2.0 / n) * grad_b;
    }
}

Matrix LinearRegression::predict(const Matrix& features) const {
    Matrix output(features.rows(), 1);
    for (std::size_t i = 0; i < features.rows(); ++i) {
        output(i, 0) = dot(weights_, features.row_vector(i)) + bias_;
    }
    return output;
}

void LinearRegression::save(const std::string& path) const {
    std::ofstream out(path);
    out << learning_rate_ << ' ' << epochs_ << ' ' << bias_ << ' ' << weights_.size() << '\n';
    for (double weight : weights_) {
        out << weight << ' ';
    }
}

void LinearRegression::load(const std::string& path) {
    std::ifstream in(path);
    std::size_t size = 0;
    in >> learning_rate_ >> epochs_ >> bias_ >> size;
    weights_.assign(size, 0.0);
    for (double& weight : weights_) {
        in >> weight;
    }
}

}  // namespace ml
