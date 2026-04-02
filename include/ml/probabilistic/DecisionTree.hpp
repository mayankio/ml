#pragma once

#include <memory>

#include "ml/core/Model.hpp"

namespace ml {

class DecisionTreeClassifier : public Model {
public:
    DecisionTreeClassifier(std::size_t max_depth = 5, std::size_t min_samples_split = 2, std::size_t max_features = 0, std::uint32_t seed = 42);

    void fit(const Matrix& features, const Matrix& targets) override;
    Matrix predict(const Matrix& features) const override;
    void save(const std::string& path) const override;
    void load(const std::string& path) override;

private:
    struct Node {
        bool is_leaf = true;
        int prediction = 0;
        std::size_t feature_index = 0;
        double threshold = 0.0;
        std::unique_ptr<Node> left;
        std::unique_ptr<Node> right;
    };

    std::unique_ptr<Node> build(const Matrix& features, const Matrix& targets, std::size_t depth);
    int predict_row(const std::vector<double>& row, const Node* node) const;
    double gini(const Matrix& targets) const;

    std::size_t max_depth_;
    std::size_t min_samples_split_;
    std::size_t max_features_;
    std::uint32_t seed_;
    std::unique_ptr<Node> root_;
};

class DecisionTreeRegressor : public Model {
public:
    DecisionTreeRegressor(std::size_t max_depth = 3, std::size_t min_samples_split = 2);

    void fit(const Matrix& features, const Matrix& targets) override;
    Matrix predict(const Matrix& features) const override;
    void save(const std::string& path) const override;
    void load(const std::string& path) override;

private:
    struct Node {
        bool is_leaf = true;
        double prediction = 0.0;
        std::size_t feature_index = 0;
        double threshold = 0.0;
        std::unique_ptr<Node> left;
        std::unique_ptr<Node> right;
    };

    std::unique_ptr<Node> build(const Matrix& features, const Matrix& targets, std::size_t depth);
    double predict_row(const std::vector<double>& row, const Node* node) const;
    double variance(const Matrix& targets) const;

    std::size_t max_depth_;
    std::size_t min_samples_split_;
    std::unique_ptr<Node> root_;
};

}  // namespace ml
