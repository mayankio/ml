#include "ml/data/Split.hpp"

#include <stdexcept>

namespace ml {

namespace {

unsigned int next_random(unsigned int& state) {
    state = state * 1664525u + 1013904223u;
    return state;
}

}

MatrixSplit train_test_split(const Matrix& features, const Matrix& targets, double test_ratio, unsigned int seed) {
    if (features.rows() != targets.rows()) {
        throw std::invalid_argument("features and targets must have the same number of rows");
    }
    if (test_ratio <= 0.0 || test_ratio >= 1.0) {
        throw std::invalid_argument("test_ratio must be in (0, 1)");
    }

    std::vector<std::size_t> indices(features.rows());
    for (std::size_t i = 0; i < indices.size(); ++i) {
        indices[i] = i;
    }
    for (std::size_t i = indices.size(); i > 1; --i) {
        const std::size_t j = next_random(seed) % i;
        const std::size_t tmp = indices[i - 1];
        indices[i - 1] = indices[j];
        indices[j] = tmp;
    }

    const std::size_t test_size = static_cast<std::size_t>(features.rows() * test_ratio);
    const std::size_t train_size = features.rows() - test_size;
    MatrixSplit split{
        Matrix(train_size, features.cols()),
        Matrix(test_size, features.cols()),
        Matrix(train_size, targets.cols()),
        Matrix(test_size, targets.cols())};

    for (std::size_t i = 0; i < train_size; ++i) {
        const std::size_t idx = indices[i];
        for (std::size_t j = 0; j < features.cols(); ++j) {
            split.x_train(i, j) = features(idx, j);
        }
        for (std::size_t j = 0; j < targets.cols(); ++j) {
            split.y_train(i, j) = targets(idx, j);
        }
    }

    for (std::size_t i = 0; i < test_size; ++i) {
        const std::size_t idx = indices[train_size + i];
        for (std::size_t j = 0; j < features.cols(); ++j) {
            split.x_test(i, j) = features(idx, j);
        }
        for (std::size_t j = 0; j < targets.cols(); ++j) {
            split.y_test(i, j) = targets(idx, j);
        }
    }

    return split;
}

}  // namespace ml
