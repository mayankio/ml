#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <vector>

#include "ml/core/Matrix.hpp"

inline void assert_close(double actual, double expected, double tolerance) {
    assert(std::fabs(actual - expected) <= tolerance);
}

inline void assert_probability_rows(const ml::Matrix& probabilities, std::size_t expected_cols, double tolerance = 1e-6) {
    assert(probabilities.cols() == expected_cols);
    for (std::size_t i = 0; i < probabilities.rows(); ++i) {
        double sum = 0.0;
        for (std::size_t j = 0; j < probabilities.cols(); ++j) {
            assert(probabilities(i, j) >= -tolerance);
            assert(probabilities(i, j) <= 1.0 + tolerance);
            sum += probabilities(i, j);
        }
        assert_close(sum, 1.0, tolerance);
    }
}

inline void assert_labels_in_set(const ml::Matrix& predictions, const std::vector<int>& labels) {
    for (std::size_t i = 0; i < predictions.rows(); ++i) {
        const int prediction = static_cast<int>(std::llround(predictions(i, 0)));
        assert(std::find(labels.begin(), labels.end(), prediction) != labels.end());
    }
}
