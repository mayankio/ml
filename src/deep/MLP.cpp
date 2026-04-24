#include "ml/deep/MLP.hpp"

#include <fstream>
#include <sstream>

#include "ml/core/ClassificationUtils.hpp"

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
    const ClassificationTargetInfo target_info = parse_classification_targets(features, targets);
    classes_ = target_info.classes;
    const std::size_t output_dim = output_dimension_for_classes(classes_);
    if (w2_.rows() != hidden_dim_ || w2_.cols() != output_dim) {
        w2_ = Matrix::random(hidden_dim_, output_dim, -0.5, 0.5, 9);
    }
    if (b1_.size() != hidden_dim_) {
        b1_.assign(hidden_dim_, 0.0);
    }
    if (b2_.size() != output_dim) {
        b2_.assign(output_dim, 0.0);
    }

    for (std::size_t epoch = 0; epoch < epochs_; ++epoch) {
        for (std::size_t i = 0; i < features.rows(); ++i) {
            Matrix x = features.row(i);
            Matrix z1 = add_row_vector(matmul(x, w1_), b1_);
            Matrix a1 = z1.apply([](double value) { return sigmoid(value); });
            Matrix z2 = add_row_vector(matmul(a1, w2_), b2_);
            std::vector<double> hidden_grads(hidden_dim_, 0.0);

            if (is_binary_classes(classes_)) {
                const double y_hat = sigmoid(z2(0, 0));
                const double dz2 = y_hat - (target_info.indices[i] == 1 ? 1.0 : 0.0);
                for (std::size_t h = 0; h < hidden_dim_; ++h) {
                    hidden_grads[h] = dz2 * w2_(h, 0) * a1(0, h) * (1.0 - a1(0, h));
                }
                for (std::size_t h = 0; h < hidden_dim_; ++h) {
                    w2_(h, 0) -= learning_rate_ * a1(0, h) * dz2;
                }
                b2_[0] -= learning_rate_ * dz2;
            } else {
                const std::vector<double> probabilities = softmax(z2.row_vector(0));
                std::vector<double> dz2(classes_.size(), 0.0);
                for (std::size_t cls = 0; cls < classes_.size(); ++cls) {
                    dz2[cls] = probabilities[cls] - (target_info.indices[i] == cls ? 1.0 : 0.0);
                }
                for (std::size_t h = 0; h < hidden_dim_; ++h) {
                    double grad = 0.0;
                    for (std::size_t cls = 0; cls < classes_.size(); ++cls) {
                        grad += dz2[cls] * w2_(h, cls);
                        w2_(h, cls) -= learning_rate_ * a1(0, h) * dz2[cls];
                    }
                    hidden_grads[h] = grad * a1(0, h) * (1.0 - a1(0, h));
                }
                for (std::size_t cls = 0; cls < classes_.size(); ++cls) {
                    b2_[cls] -= learning_rate_ * dz2[cls];
                }
            }

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
    if (classes_.empty()) {
        throw std::logic_error("MLPClassifier must be fit before predict_proba");
    }
    Matrix output(features.rows(), output_dimension_for_classes(classes_));
    for (std::size_t i = 0; i < features.rows(); ++i) {
        Matrix a1 = add_row_vector(matmul(features.row(i), w1_), b1_).apply([](double value) { return sigmoid(value); });
        Matrix z2 = add_row_vector(matmul(a1, w2_), b2_);
        if (is_binary_classes(classes_)) {
            output(i, 0) = sigmoid(z2(0, 0));
            continue;
        }

        const std::vector<double> probabilities = softmax(z2.row_vector(0));
        for (std::size_t cls = 0; cls < classes_.size(); ++cls) {
            output(i, cls) = probabilities[cls];
        }
    }
    return output;
}

Matrix MLPClassifier::predict(const Matrix& features) const {
    const Matrix probabilities = predict_proba(features);
    if (is_binary_classes(classes_)) {
        return decode_binary_predictions(classes_, probabilities, 0.5);
    }
    return decode_multiclass_predictions(classes_, probabilities);
}

const std::vector<int>& MLPClassifier::classes() const {
    return classes_;
}

std::size_t MLPClassifier::num_classes() const {
    return classes_.size();
}

void MLPClassifier::save(const std::string& path) const {
    std::ofstream out(path);
    out << "v2\n";
    out << input_dim_ << ' ' << hidden_dim_ << ' ' << learning_rate_ << ' ' << epochs_ << '\n';
    save_vector(out, classes_);
    save_vector(out, b1_);
    save_vector(out, b2_);
    save_matrix(out, w1_);
    save_matrix(out, w2_);
}

void MLPClassifier::load(const std::string& path) {
    std::ifstream in(path);
    std::string version;
    in >> version;
    if (version == "v2") {
        in >> input_dim_ >> hidden_dim_ >> learning_rate_ >> epochs_;
        classes_ = load_vector<int>(in);
        b1_ = load_vector<double>(in);
        b2_ = load_vector<double>(in);
        w1_ = load_matrix(in);
        w2_ = load_matrix(in);
        return;
    }

    std::istringstream header(version);
    header >> input_dim_;
    in >> hidden_dim_ >> learning_rate_ >> epochs_;
    w1_ = load_matrix(in);
    w2_ = load_matrix(in);
    classes_ = {0, 1};
    b1_.assign(hidden_dim_, 0.0);
    b2_.assign(1, 0.0);
}

}  // namespace ml
