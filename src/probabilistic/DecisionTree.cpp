#include "ml/probabilistic/DecisionTree.hpp"

#include <algorithm>
#include <fstream>
#include <limits>
#include <map>
#include <numeric>
#include <random>

namespace ml {

namespace {

Matrix gather_rows(const Matrix& matrix, const std::vector<std::size_t>& indices) {
    Matrix result(indices.size(), matrix.cols());
    for (std::size_t i = 0; i < indices.size(); ++i) {
        for (std::size_t j = 0; j < matrix.cols(); ++j) {
            result(i, j) = matrix(indices[i], j);
        }
    }
    return result;
}

int majority_label(const Matrix& targets) {
    std::map<int, int> counts;
    for (std::size_t i = 0; i < targets.rows(); ++i) {
        counts[static_cast<int>(targets(i, 0))]++;
    }
    return std::max_element(counts.begin(), counts.end(),
                            [](const auto& lhs, const auto& rhs) { return lhs.second < rhs.second; })
        ->first;
}

double mean_target(const Matrix& targets) {
    double sum = 0.0;
    for (std::size_t i = 0; i < targets.rows(); ++i) {
        sum += targets(i, 0);
    }
    return sum / static_cast<double>(targets.rows());
}

}  // namespace

DecisionTreeClassifier::DecisionTreeClassifier(std::size_t max_depth, std::size_t min_samples_split, std::size_t max_features, std::uint32_t seed)
    : max_depth_(max_depth), min_samples_split_(min_samples_split), max_features_(max_features), seed_(seed) {}

double DecisionTreeClassifier::gini(const Matrix& targets) const {
    std::map<int, int> counts;
    for (std::size_t i = 0; i < targets.rows(); ++i) {
        counts[static_cast<int>(targets(i, 0))]++;
    }
    double impurity = 1.0;
    for (const auto& [label, count] : counts) {
        const double p = static_cast<double>(count) / static_cast<double>(targets.rows());
        impurity -= p * p;
    }
    return impurity;
}

std::unique_ptr<DecisionTreeClassifier::Node> DecisionTreeClassifier::build(const Matrix& features, const Matrix& targets, std::size_t depth) {
    auto node = std::make_unique<Node>();
    node->prediction = majority_label(targets);
    if (depth >= max_depth_ || targets.rows() < min_samples_split_ || gini(targets) < 1e-9) {
        return node;
    }

    std::vector<std::size_t> feature_indices(features.cols());
    std::iota(feature_indices.begin(), feature_indices.end(), 0);
    std::mt19937 rng(seed_ + static_cast<std::uint32_t>(depth));
    std::shuffle(feature_indices.begin(), feature_indices.end(), rng);
    if (max_features_ > 0 && max_features_ < feature_indices.size()) {
        feature_indices.resize(max_features_);
    }

    double best_score = std::numeric_limits<double>::infinity();
    std::size_t best_feature = 0;
    double best_threshold = 0.0;
    std::vector<std::size_t> best_left;
    std::vector<std::size_t> best_right;

    for (std::size_t feature : feature_indices) {
        for (std::size_t i = 0; i < features.rows(); ++i) {
            const double threshold = features(i, feature);
            std::vector<std::size_t> left_indices;
            std::vector<std::size_t> right_indices;
            for (std::size_t row = 0; row < features.rows(); ++row) {
                if (features(row, feature) <= threshold) {
                    left_indices.push_back(row);
                } else {
                    right_indices.push_back(row);
                }
            }
            if (left_indices.empty() || right_indices.empty()) {
                continue;
            }
            const Matrix left_targets = gather_rows(targets, left_indices);
            const Matrix right_targets = gather_rows(targets, right_indices);
            const double score = (left_targets.rows() * gini(left_targets) + right_targets.rows() * gini(right_targets)) /
                                 static_cast<double>(targets.rows());
            if (score < best_score) {
                best_score = score;
                best_feature = feature;
                best_threshold = threshold;
                best_left = left_indices;
                best_right = right_indices;
            }
        }
    }

    if (best_left.empty() || best_right.empty()) {
        return node;
    }

    node->is_leaf = false;
    node->feature_index = best_feature;
    node->threshold = best_threshold;
    node->left = build(gather_rows(features, best_left), gather_rows(targets, best_left), depth + 1);
    node->right = build(gather_rows(features, best_right), gather_rows(targets, best_right), depth + 1);
    return node;
}

void DecisionTreeClassifier::fit(const Matrix& features, const Matrix& targets) {
    root_ = build(features, targets, 0);
}

int DecisionTreeClassifier::predict_row(const std::vector<double>& row, const Node* node) const {
    if (node->is_leaf) {
        return node->prediction;
    }
    if (row[node->feature_index] <= node->threshold) {
        return predict_row(row, node->left.get());
    }
    return predict_row(row, node->right.get());
}

Matrix DecisionTreeClassifier::predict(const Matrix& features) const {
    Matrix output(features.rows(), 1);
    for (std::size_t i = 0; i < features.rows(); ++i) {
        output(i, 0) = predict_row(features.row_vector(i), root_.get());
    }
    return output;
}

void DecisionTreeClassifier::save(const std::string& path) const {
    std::ofstream out(path);
    out << max_depth_ << ' ' << min_samples_split_ << ' ' << max_features_ << ' ' << seed_ << '\n';
}

void DecisionTreeClassifier::load(const std::string& path) {
    std::ifstream in(path);
    in >> max_depth_ >> min_samples_split_ >> max_features_ >> seed_;
}

DecisionTreeRegressor::DecisionTreeRegressor(std::size_t max_depth, std::size_t min_samples_split)
    : max_depth_(max_depth), min_samples_split_(min_samples_split) {}

double DecisionTreeRegressor::variance(const Matrix& targets) const {
    const double mean = mean_target(targets);
    double sum = 0.0;
    for (std::size_t i = 0; i < targets.rows(); ++i) {
        const double diff = targets(i, 0) - mean;
        sum += diff * diff;
    }
    return sum / static_cast<double>(targets.rows());
}

std::unique_ptr<DecisionTreeRegressor::Node> DecisionTreeRegressor::build(const Matrix& features, const Matrix& targets, std::size_t depth) {
    auto node = std::make_unique<Node>();
    node->prediction = mean_target(targets);
    if (depth >= max_depth_ || targets.rows() < min_samples_split_ || variance(targets) < 1e-9) {
        return node;
    }

    double best_score = std::numeric_limits<double>::infinity();
    std::size_t best_feature = 0;
    double best_threshold = 0.0;
    std::vector<std::size_t> best_left;
    std::vector<std::size_t> best_right;

    for (std::size_t feature = 0; feature < features.cols(); ++feature) {
        for (std::size_t i = 0; i < features.rows(); ++i) {
            const double threshold = features(i, feature);
            std::vector<std::size_t> left_indices;
            std::vector<std::size_t> right_indices;
            for (std::size_t row = 0; row < features.rows(); ++row) {
                if (features(row, feature) <= threshold) {
                    left_indices.push_back(row);
                } else {
                    right_indices.push_back(row);
                }
            }
            if (left_indices.empty() || right_indices.empty()) {
                continue;
            }
            const Matrix left_targets = gather_rows(targets, left_indices);
            const Matrix right_targets = gather_rows(targets, right_indices);
            const double score = (left_targets.rows() * variance(left_targets) + right_targets.rows() * variance(right_targets)) /
                                 static_cast<double>(targets.rows());
            if (score < best_score) {
                best_score = score;
                best_feature = feature;
                best_threshold = threshold;
                best_left = left_indices;
                best_right = right_indices;
            }
        }
    }

    if (best_left.empty() || best_right.empty()) {
        return node;
    }

    node->is_leaf = false;
    node->feature_index = best_feature;
    node->threshold = best_threshold;
    node->left = build(gather_rows(features, best_left), gather_rows(targets, best_left), depth + 1);
    node->right = build(gather_rows(features, best_right), gather_rows(targets, best_right), depth + 1);
    return node;
}

void DecisionTreeRegressor::fit(const Matrix& features, const Matrix& targets) {
    root_ = build(features, targets, 0);
}

double DecisionTreeRegressor::predict_row(const std::vector<double>& row, const Node* node) const {
    if (node->is_leaf) {
        return node->prediction;
    }
    if (row[node->feature_index] <= node->threshold) {
        return predict_row(row, node->left.get());
    }
    return predict_row(row, node->right.get());
}

Matrix DecisionTreeRegressor::predict(const Matrix& features) const {
    Matrix output(features.rows(), 1);
    for (std::size_t i = 0; i < features.rows(); ++i) {
        output(i, 0) = predict_row(features.row_vector(i), root_.get());
    }
    return output;
}

void DecisionTreeRegressor::save(const std::string& path) const {
    std::ofstream out(path);
    out << max_depth_ << ' ' << min_samples_split_ << '\n';
}

void DecisionTreeRegressor::load(const std::string& path) {
    std::ifstream in(path);
    in >> max_depth_ >> min_samples_split_;
}

}  // namespace ml
