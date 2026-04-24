#pragma once

#include <vector>

#include "ml/core/Model.hpp"

namespace ml {

class LogisticRegression : public Model {
public:
    LogisticRegression(double learning_rate = 0.1, std::size_t epochs = 2000);

    void fit(const Matrix& features, const Matrix& targets) override;
    Matrix predict(const Matrix& features) const override;
    Matrix predict_proba(const Matrix& features) const;
    [[nodiscard]] const std::vector<int>& classes() const;
    [[nodiscard]] std::size_t num_classes() const;
    void save(const std::string& path) const override;
    void load(const std::string& path) override;

private:
    double learning_rate_;
    std::size_t epochs_;
    Matrix weights_;
    std::vector<double> bias_;
    std::vector<int> classes_;
};

}  // namespace ml
