#pragma once

#include "ml/core/Model.hpp"

namespace ml {

class SelfAttention : public Model {
public:
    SelfAttention(std::size_t sequence_length, std::size_t embedding_dim, std::size_t projection_dim);

    void fit(const Matrix& features, const Matrix& targets) override;
    Matrix predict(const Matrix& features) const override;
    void save(const std::string& path) const override;
    void load(const std::string& path) override;

private:
    Matrix forward(const Matrix& features) const;

    std::size_t sequence_length_;
    std::size_t embedding_dim_;
    std::size_t projection_dim_;
    Matrix wq_;
    Matrix wk_;
    Matrix wv_;
    Matrix wo_;
};

}  // namespace ml
