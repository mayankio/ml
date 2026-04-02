#pragma once

#include "ml/core/Model.hpp"

namespace ml {

class LinearSVM : public Model {
public:
    LinearSVM(double learning_rate = 0.01, std::size_t epochs = 2000, double c = 1.0, bool hard_margin = false);

    void fit(const Matrix& features, const Matrix& targets) override;
    Matrix predict(const Matrix& features) const override;
    Matrix decision_function(const Matrix& features) const;
    void save(const std::string& path) const override;
    void load(const std::string& path) override;

private:
    double learning_rate_;
    std::size_t epochs_;
    double c_;
    bool hard_margin_;
    std::vector<double> weights_;
    double bias_ = 0.0;
};

}  // namespace ml
