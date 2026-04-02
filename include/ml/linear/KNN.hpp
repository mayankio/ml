#pragma once

#include "ml/core/Model.hpp"

namespace ml {

class KNNClassifier : public Model {
public:
    explicit KNNClassifier(std::size_t k = 3);

    void fit(const Matrix& features, const Matrix& targets) override;
    Matrix predict(const Matrix& features) const override;
    void save(const std::string& path) const override;
    void load(const std::string& path) override;

private:
    std::size_t k_;
    Matrix train_features_;
    Matrix train_targets_;
};

}  // namespace ml
