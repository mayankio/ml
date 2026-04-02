#pragma once

#include "ml/core/Model.hpp"

namespace ml {

class PCA : public Model {
public:
    explicit PCA(std::size_t n_components = 2, std::size_t power_iterations = 100);

    void fit(const Matrix& features, const Matrix& targets) override;
    Matrix predict(const Matrix& features) const override;
    Matrix transform(const Matrix& features) const;
    const Matrix& components() const;
    void save(const std::string& path) const override;
    void load(const std::string& path) override;

private:
    std::size_t n_components_;
    std::size_t power_iterations_;
    std::vector<double> means_;
    Matrix components_;
};

}  // namespace ml
