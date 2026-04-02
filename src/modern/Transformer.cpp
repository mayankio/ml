#include "ml/modern/Transformer.hpp"

#include <fstream>

namespace ml {

TransformerClassifier::TransformerClassifier(std::size_t sequence_length, std::size_t embedding_dim, std::size_t projection_dim, std::size_t hidden_dim, double learning_rate, std::size_t epochs)
    : sequence_length_(sequence_length),
      embedding_dim_(embedding_dim),
      projection_dim_(projection_dim),
      hidden_dim_(hidden_dim),
      learning_rate_(learning_rate),
      epochs_(epochs),
      attention_(sequence_length, embedding_dim, projection_dim),
      ff1_(Matrix::random(embedding_dim, hidden_dim, -0.2, 0.2, 61)),
      ff2_(Matrix::random(hidden_dim, embedding_dim, -0.2, 0.2, 62)),
      b1_(hidden_dim, 0.0),
      b2_(embedding_dim, 0.0),
      out_(Matrix::random(embedding_dim, 1, -0.2, 0.2, 63)) {}

Matrix TransformerClassifier::encode(const Matrix& features) const {
    Matrix attended = attention_.predict(features);
    Matrix ff_hidden = add_row_vector(matmul(attended, ff1_), b1_).apply([](double value) { return relu(value); });
    Matrix ff_output = add_row_vector(matmul(ff_hidden, ff2_), b2_);
    Matrix pooled(1, embedding_dim_, 0.0);
    for (std::size_t i = 0; i < ff_output.rows(); ++i) {
        for (std::size_t j = 0; j < ff_output.cols(); ++j) {
            pooled(0, j) += ff_output(i, j);
        }
    }
    for (std::size_t j = 0; j < pooled.cols(); ++j) {
        pooled(0, j) /= static_cast<double>(ff_output.rows());
    }
    return pooled;
}

void TransformerClassifier::fit(const Matrix& features, const Matrix& targets) {
    for (std::size_t epoch = 0; epoch < epochs_; ++epoch) {
        for (std::size_t sample = 0; sample < features.rows(); ++sample) {
            Matrix sequence(sequence_length_, embedding_dim_);
            for (std::size_t t = 0; t < sequence_length_; ++t) {
                for (std::size_t d = 0; d < embedding_dim_; ++d) {
                    sequence(t, d) = features(sample, t * embedding_dim_ + d);
                }
            }
            Matrix pooled = encode(sequence);
            const double prediction = sigmoid(matmul(pooled, out_)(0, 0) + out_bias_);
            const double error = prediction - targets(sample, 0);
            for (std::size_t j = 0; j < embedding_dim_; ++j) {
                out_(j, 0) -= learning_rate_ * error * pooled(0, j);
            }
            out_bias_ -= learning_rate_ * error;
        }
    }
}

Matrix TransformerClassifier::predict_proba(const Matrix& features) const {
    Matrix output(features.rows(), 1);
    for (std::size_t sample = 0; sample < features.rows(); ++sample) {
        Matrix sequence(sequence_length_, embedding_dim_);
        for (std::size_t t = 0; t < sequence_length_; ++t) {
            for (std::size_t d = 0; d < embedding_dim_; ++d) {
                sequence(t, d) = features(sample, t * embedding_dim_ + d);
            }
        }
        Matrix pooled = encode(sequence);
        output(sample, 0) = sigmoid(matmul(pooled, out_)(0, 0) + out_bias_);
    }
    return output;
}

Matrix TransformerClassifier::predict(const Matrix& features) const {
    Matrix probabilities = predict_proba(features);
    for (std::size_t i = 0; i < probabilities.rows(); ++i) {
        probabilities(i, 0) = probabilities(i, 0) >= 0.5 ? 1.0 : 0.0;
    }
    return probabilities;
}

void TransformerClassifier::save(const std::string& path) const {
    std::ofstream out(path);
    out << sequence_length_ << ' ' << embedding_dim_ << ' ' << projection_dim_ << ' ' << hidden_dim_ << ' ' << learning_rate_ << ' ' << epochs_ << ' ' << out_bias_ << '\n';
    save_matrix(out, ff1_);
    save_matrix(out, ff2_);
    save_matrix(out, out_);
}

void TransformerClassifier::load(const std::string& path) {
    std::ifstream in(path);
    in >> sequence_length_ >> embedding_dim_ >> projection_dim_ >> hidden_dim_ >> learning_rate_ >> epochs_ >> out_bias_;
    ff1_ = load_matrix(in);
    ff2_ = load_matrix(in);
    out_ = load_matrix(in);
    b1_.assign(hidden_dim_, 0.0);
    b2_.assign(embedding_dim_, 0.0);
}

}  // namespace ml
