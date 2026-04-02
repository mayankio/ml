#include "ml/deep/MLP.hpp"

#include <fstream>

namespace ml {

MLPClassifier::MLPClassifier(std::size_t input_dim, std::size_t hidden_dim, double learning_rate, std::size_t epochs)
    : input_dim_(input_dim),
      hidden_dim_(hidden_dim),
      learning_rate_(learning_rate),
      epochs_(epochs),
      w1_(Matrix::random(input_dim, hidden_dim, -0.5, 0.5, 7)),
      w2_(Matrix::random(hidden_dim, 1, -0.5, 0.5, 9)),
      b1_(hidden_dim, 0.0),
      b2_(1, 0.0) {}

void MLPClassifier::fit(const Matrix& features, const Matrix& targets) {
    for (std::size_t epoch = 0; epoch < epochs_; ++epoch) {
        for (std::size_t i = 0; i < features.rows(); ++i) {
            Matrix x = features.row(i);
            Matrix z1 = add_row_vector(matmul(x, w1_), b1_);
            Matrix a1 = z1.apply([](double value) { return sigmoid(value); });
            Matrix z2 = add_row_vector(matmul(a1, w2_), b2_);
            const double y_hat = sigmoid(z2(0, 0));
            const double dz2 = y_hat - targets(i, 0);
            std::vector<double> hidden_grads(hidden_dim_, 0.0);
            for (std::size_t h = 0; h < hidden_dim_; ++h) {
                hidden_grads[h] = dz2 * w2_(h, 0) * a1(0, h) * (1.0 - a1(0, h));
            }
            for (std::size_t h = 0; h < hidden_dim_; ++h) {
                w2_(h, 0) -= learning_rate_ * a1(0, h) * dz2;
            }
            b2_[0] -= learning_rate_ * dz2;
            for (std::size_t h = 0; h < hidden_dim_; ++h) {
                for (std::size_t j = 0; j < input_dim_; ++j) {
                    w1_(j, h) -= learning_rate_ * x(0, j) * hidden_grads[h];
                }
                b1_[h] -= learning_rate_ * hidden_grads[h];
            }
        }
    }
}

Matrix MLPClassifier::predict_proba(const Matrix& features) const {
    Matrix output(features.rows(), 1);
    for (std::size_t i = 0; i < features.rows(); ++i) {
        Matrix a1 = add_row_vector(matmul(features.row(i), w1_), b1_).apply([](double value) { return sigmoid(value); });
        output(i, 0) = sigmoid(add_row_vector(matmul(a1, w2_), b2_)(0, 0));
    }
    return output;
}

Matrix MLPClassifier::predict(const Matrix& features) const {
    Matrix probabilities = predict_proba(features);
    for (std::size_t i = 0; i < probabilities.rows(); ++i) {
        probabilities(i, 0) = probabilities(i, 0) >= 0.5 ? 1.0 : 0.0;
    }
    return probabilities;
}

void MLPClassifier::save(const std::string& path) const {
    std::ofstream out(path);
    out << input_dim_ << ' ' << hidden_dim_ << ' ' << learning_rate_ << ' ' << epochs_ << '\n';
    save_matrix(out, w1_);
    save_matrix(out, w2_);
}

void MLPClassifier::load(const std::string& path) {
    std::ifstream in(path);
    in >> input_dim_ >> hidden_dim_ >> learning_rate_ >> epochs_;
    w1_ = load_matrix(in);
    w2_ = load_matrix(in);
    b1_.assign(hidden_dim_, 0.0);
    b2_.assign(1, 0.0);
}

}  // namespace ml
