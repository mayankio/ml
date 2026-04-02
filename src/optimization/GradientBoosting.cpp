#include "ml/optimization/GradientBoosting.hpp"

#include <fstream>

namespace ml {

GradientBoostingRegressor::GradientBoostingRegressor(std::size_t n_estimators, double learning_rate, std::size_t max_depth)
    : n_estimators_(n_estimators), learning_rate_(learning_rate), max_depth_(max_depth) {}

void GradientBoostingRegressor::fit(const Matrix& features, const Matrix& targets) {
    trees_.clear();
    init_prediction_ = 0.0;
    for (std::size_t i = 0; i < targets.rows(); ++i) {
        init_prediction_ += targets(i, 0);
    }
    init_prediction_ /= static_cast<double>(targets.rows());
    Matrix current(features.rows(), 1, init_prediction_);
    for (std::size_t estimator = 0; estimator < n_estimators_; ++estimator) {
        Matrix residuals(targets.rows(), 1);
        for (std::size_t i = 0; i < targets.rows(); ++i) {
            residuals(i, 0) = targets(i, 0) - current(i, 0);
        }
        DecisionTreeRegressor tree(max_depth_, 2);
        tree.fit(features, residuals);
        Matrix update = tree.predict(features);
        for (std::size_t i = 0; i < current.rows(); ++i) {
            current(i, 0) += learning_rate_ * update(i, 0);
        }
        trees_.push_back(std::move(tree));
    }
}

Matrix GradientBoostingRegressor::predict(const Matrix& features) const {
    Matrix output(features.rows(), 1, init_prediction_);
    for (const auto& tree : trees_) {
        Matrix update = tree.predict(features);
        for (std::size_t i = 0; i < output.rows(); ++i) {
            output(i, 0) += learning_rate_ * update(i, 0);
        }
    }
    return output;
}

void GradientBoostingRegressor::save(const std::string& path) const {
    std::ofstream out(path);
    out << n_estimators_ << ' ' << learning_rate_ << ' ' << max_depth_ << ' ' << init_prediction_ << '\n';
}

void GradientBoostingRegressor::load(const std::string& path) {
    std::ifstream in(path);
    in >> n_estimators_ >> learning_rate_ >> max_depth_ >> init_prediction_;
}

}  // namespace ml
