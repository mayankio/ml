#pragma once

#include <vector>

#include "ml/core/Model.hpp"

namespace ml {

class SimpleCNN : public Model {
public:
    SimpleCNN(std::size_t image_height, std::size_t image_width, std::size_t num_filters = 2, std::size_t kernel_size = 2, double learning_rate = 0.05, std::size_t epochs = 500);

    void fit(const Matrix& features, const Matrix& targets) override;
    Matrix predict(const Matrix& features) const override;
    Matrix predict_proba(const Matrix& features) const;
    [[nodiscard]] const std::vector<int>& classes() const;
    [[nodiscard]] std::size_t num_classes() const;
    void save(const std::string& path) const override;
    void load(const std::string& path) override;

private:
    std::vector<double> forward_single(const std::vector<double>& input, std::vector<double>* hidden = nullptr) const;
    std::size_t image_height_;
    std::size_t image_width_;
    std::size_t num_filters_;
    std::size_t kernel_size_;
    double learning_rate_;
    std::size_t epochs_;
    std::vector<Matrix> filters_;
    Matrix dense_weights_;
    std::vector<double> dense_bias_;
    std::vector<int> classes_;
};

}  // namespace ml
