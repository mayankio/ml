#include "ml/linear/KNN.hpp"

#include <algorithm>
#include <fstream>
#include <map>

namespace ml {

KNNClassifier::KNNClassifier(std::size_t k) : k_(k) {}

void KNNClassifier::fit(const Matrix& features, const Matrix& targets) {
    train_features_ = features;
    train_targets_ = targets;
}

Matrix KNNClassifier::predict(const Matrix& features) const {
    Matrix output(features.rows(), 1);
    for (std::size_t i = 0; i < features.rows(); ++i) {
        std::vector<std::pair<double, int>> distances;
        const auto row = features.row_vector(i);
        for (std::size_t j = 0; j < train_features_.rows(); ++j) {
            distances.emplace_back(euclidean_distance(row, train_features_.row_vector(j)), static_cast<int>(train_targets_(j, 0)));
        }
        std::nth_element(distances.begin(), distances.begin() + std::min(k_, distances.size()) - 1, distances.end(),
                         [](const auto& lhs, const auto& rhs) { return lhs.first < rhs.first; });
        std::map<int, int> votes;
        for (std::size_t neighbor = 0; neighbor < std::min(k_, distances.size()); ++neighbor) {
            votes[distances[neighbor].second]++;
        }
        output(i, 0) = std::max_element(votes.begin(), votes.end(),
                                        [](const auto& lhs, const auto& rhs) { return lhs.second < rhs.second; })
                           ->first;
    }
    return output;
}

void KNNClassifier::save(const std::string& path) const {
    std::ofstream out(path);
    out << k_ << '\n';
    save_matrix(out, train_features_);
    save_matrix(out, train_targets_);
}

void KNNClassifier::load(const std::string& path) {
    std::ifstream in(path);
    in >> k_;
    train_features_ = load_matrix(in);
    train_targets_ = load_matrix(in);
}

}  // namespace ml
