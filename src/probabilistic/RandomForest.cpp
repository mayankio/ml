#include "ml/probabilistic/RandomForest.hpp"

#include <cmath>
#include <fstream>
#include <map>
#include <random>

namespace ml {

RandomForestClassifier::RandomForestClassifier(std::size_t n_estimators, std::size_t max_depth, std::size_t min_samples_split, std::uint32_t seed)
    : n_estimators_(n_estimators), max_depth_(max_depth), min_samples_split_(min_samples_split), seed_(seed) {}

void RandomForestClassifier::fit(const Matrix& features, const Matrix& targets) {
    trees_.clear();
    const std::size_t max_features = std::max<std::size_t>(1, static_cast<std::size_t>(std::sqrt(static_cast<double>(features.cols()))));
    std::mt19937 rng(seed_);
    std::uniform_int_distribution<std::size_t> dist(0, features.rows() - 1);
    for (std::size_t tree_index = 0; tree_index < n_estimators_; ++tree_index) {
        Matrix boot_features(features.rows(), features.cols());
        Matrix boot_targets(targets.rows(), 1);
        for (std::size_t i = 0; i < features.rows(); ++i) {
            const std::size_t sample = dist(rng);
            for (std::size_t j = 0; j < features.cols(); ++j) {
                boot_features(i, j) = features(sample, j);
            }
            boot_targets(i, 0) = targets(sample, 0);
        }
        DecisionTreeClassifier tree(max_depth_, min_samples_split_, max_features, seed_ + static_cast<std::uint32_t>(tree_index));
        tree.fit(boot_features, boot_targets);
        trees_.push_back(std::move(tree));
    }
}

Matrix RandomForestClassifier::predict(const Matrix& features) const {
    Matrix output(features.rows(), 1);
    for (std::size_t i = 0; i < features.rows(); ++i) {
        std::map<int, int> votes;
        const Matrix sample = features.row(i);
        for (const auto& tree : trees_) {
            votes[static_cast<int>(tree.predict(sample)(0, 0))]++;
        }
        output(i, 0) = std::max_element(votes.begin(), votes.end(),
                                        [](const auto& lhs, const auto& rhs) { return lhs.second < rhs.second; })
                           ->first;
    }
    return output;
}

void RandomForestClassifier::save(const std::string& path) const {
    std::ofstream out(path);
    out << n_estimators_ << ' ' << max_depth_ << ' ' << min_samples_split_ << ' ' << seed_ << '\n';
}

void RandomForestClassifier::load(const std::string& path) {
    std::ifstream in(path);
    in >> n_estimators_ >> max_depth_ >> min_samples_split_ >> seed_;
}

}  // namespace ml
