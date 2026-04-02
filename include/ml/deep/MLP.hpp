#pragma once

#include "ml/core/Model.hpp"

namespace ml {

class MLPClassifier : public Model {
public:
    MLPClassifier(std::size_t input_dim, std::size_t hidden_dim, double learning_rate = 0.1, std::size_t epochs = 2000);

    void fit(const Matrix& features, const Matrix& targets) override;
    Matrix predict(const Matrix& features) const override;
    Matrix predict_proba(const Matrix& features) const;
    void save(const std::string& path) const override;
    void load(const std::string& path) override;

private:
    std::size_t input_dim_;
    std::size_t hidden_dim_;
    double learning_rate_;
    std::size_t epochs_;
    Matrix w1_;
    Matrix w2_;
    std::vector<double> b1_;
    std::vector<double> b2_;
};

}  // namespace ml
