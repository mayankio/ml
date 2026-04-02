#pragma once

#include "ml/core/Model.hpp"

namespace ml {

class LinearRegression : public Model {
public:
    LinearRegression(double learning_rate = 0.01, std::size_t epochs = 2000);

    void fit(const Matrix& features, const Matrix& targets) override;
    Matrix predict(const Matrix& features) const override;
    void save(const std::string& path) const override;
    void load(const std::string& path) override;

private:
    double learning_rate_;
    std::size_t epochs_;
    std::vector<double> weights_;
    double bias_ = 0.0;
};

}  // namespace ml
