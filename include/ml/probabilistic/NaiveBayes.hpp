#pragma once

#include <map>

#include "ml/core/Model.hpp"

namespace ml {

class GaussianNaiveBayes : public Model {
public:
    void fit(const Matrix& features, const Matrix& targets) override;
    Matrix predict(const Matrix& features) const override;
    void save(const std::string& path) const override;
    void load(const std::string& path) override;

private:
    std::vector<int> classes_;
    std::map<int, std::vector<double>> means_;
    std::map<int, std::vector<double>> variances_;
    std::map<int, double> priors_;
};

}  // namespace ml
