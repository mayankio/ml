#pragma once

#include "ml/core/Model.hpp"
#include "ml/modern/SelfAttention.hpp"

namespace ml {

class TransformerClassifier : public Model {
public:
    TransformerClassifier(std::size_t sequence_length, std::size_t embedding_dim, std::size_t projection_dim, std::size_t hidden_dim, double learning_rate = 0.05, std::size_t epochs = 500);

    void fit(const Matrix& features, const Matrix& targets) override;
    Matrix predict(const Matrix& features) const override;
    Matrix predict_proba(const Matrix& features) const;
    void save(const std::string& path) const override;
    void load(const std::string& path) override;

private:
    Matrix encode(const Matrix& features) const;

    std::size_t sequence_length_;
    std::size_t embedding_dim_;
    std::size_t projection_dim_;
    std::size_t hidden_dim_;
    double learning_rate_;
    std::size_t epochs_;
    SelfAttention attention_;
    Matrix ff1_;
    Matrix ff2_;
    std::vector<double> b1_;
    std::vector<double> b2_;
    Matrix out_;
    double out_bias_ = 0.0;
};

}  // namespace ml
