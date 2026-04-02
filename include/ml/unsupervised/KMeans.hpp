#pragma once

#include "ml/core/Model.hpp"

namespace ml {

class KMeans : public Model {
public:
    KMeans(std::size_t n_clusters = 2, std::size_t max_iters = 100, std::uint32_t seed = 42);

    void fit(const Matrix& features, const Matrix& targets) override;
    Matrix predict(const Matrix& features) const override;
    const Matrix& centroids() const;
    void save(const std::string& path) const override;
    void load(const std::string& path) override;

private:
    std::size_t n_clusters_;
    std::size_t max_iters_;
    std::uint32_t seed_;
    Matrix centroids_;
};

}  // namespace ml
