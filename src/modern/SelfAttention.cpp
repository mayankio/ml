#include "ml/modern/SelfAttention.hpp"

#include <fstream>

namespace ml {

SelfAttention::SelfAttention(std::size_t sequence_length, std::size_t embedding_dim, std::size_t projection_dim)
    : sequence_length_(sequence_length),
      embedding_dim_(embedding_dim),
      projection_dim_(projection_dim),
      wq_(Matrix::random(embedding_dim, projection_dim, -0.2, 0.2, 51)),
      wk_(Matrix::random(embedding_dim, projection_dim, -0.2, 0.2, 52)),
      wv_(Matrix::random(embedding_dim, projection_dim, -0.2, 0.2, 53)),
      wo_(Matrix::random(projection_dim, embedding_dim, -0.2, 0.2, 54)) {}

Matrix SelfAttention::forward(const Matrix& features) const {
    Matrix q = matmul(features, wq_);
    Matrix k = matmul(features, wk_);
    Matrix v = matmul(features, wv_);
    Matrix scores(sequence_length_, sequence_length_);
    for (std::size_t i = 0; i < sequence_length_; ++i) {
        std::vector<double> row(sequence_length_);
        for (std::size_t j = 0; j < sequence_length_; ++j) {
            double score = 0.0;
            for (std::size_t d = 0; d < projection_dim_; ++d) {
                score += q(i, d) * k(j, d);
            }
            row[j] = score / std::sqrt(static_cast<double>(projection_dim_));
        }
        const auto probs = softmax(row);
        for (std::size_t j = 0; j < sequence_length_; ++j) {
            scores(i, j) = probs[j];
        }
    }
    return matmul(matmul(scores, v), wo_);
}

void SelfAttention::fit(const Matrix&, const Matrix&) {}

Matrix SelfAttention::predict(const Matrix& features) const {
    return forward(features);
}

void SelfAttention::save(const std::string& path) const {
    std::ofstream out(path);
    out << sequence_length_ << ' ' << embedding_dim_ << ' ' << projection_dim_ << '\n';
    save_matrix(out, wq_);
    save_matrix(out, wk_);
    save_matrix(out, wv_);
    save_matrix(out, wo_);
}

void SelfAttention::load(const std::string& path) {
    std::ifstream in(path);
    in >> sequence_length_ >> embedding_dim_ >> projection_dim_;
    wq_ = load_matrix(in);
    wk_ = load_matrix(in);
    wv_ = load_matrix(in);
    wo_ = load_matrix(in);
}

}  // namespace ml
