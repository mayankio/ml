#pragma once

#include "ml/core/Model.hpp"
#include "ml/probabilistic/DecisionTree.hpp"

namespace ml {

class RandomForestClassifier : public Model {
public:
    RandomForestClassifier(std::size_t n_estimators = 10, std::size_t max_depth = 5, std::size_t min_samples_split = 2, std::uint32_t seed = 42);

    void fit(const Matrix& features, const Matrix& targets) override;
    Matrix predict(const Matrix& features) const override;
    void save(const std::string& path) const override;
    void load(const std::string& path) override;

private:
    std::size_t n_estimators_;
    std::size_t max_depth_;
    std::size_t min_samples_split_;
    std::uint32_t seed_;
    std::vector<DecisionTreeClassifier> trees_;
};

}  // namespace ml
