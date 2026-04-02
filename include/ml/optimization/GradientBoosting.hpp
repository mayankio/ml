#pragma once

#include "ml/core/Model.hpp"
#include "ml/probabilistic/DecisionTree.hpp"

namespace ml {

class GradientBoostingRegressor : public Model {
public:
    GradientBoostingRegressor(std::size_t n_estimators = 20, double learning_rate = 0.1, std::size_t max_depth = 2);

    void fit(const Matrix& features, const Matrix& targets) override;
    Matrix predict(const Matrix& features) const override;
    void save(const std::string& path) const override;
    void load(const std::string& path) override;

private:
    std::size_t n_estimators_;
    double learning_rate_;
    std::size_t max_depth_;
    double init_prediction_ = 0.0;
    std::vector<DecisionTreeRegressor> trees_;
};

}  // namespace ml
