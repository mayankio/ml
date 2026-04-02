#pragma once

#include <cmath>

#include "ml/core/Matrix.hpp"

namespace ml {

inline double accuracy_score(const Matrix& predictions, const Matrix& targets) {
    std::size_t correct = 0;
    for (std::size_t i = 0; i < predictions.rows(); ++i) {
        if (std::llround(predictions(i, 0)) == std::llround(targets(i, 0))) {
            ++correct;
        }
    }
    return static_cast<double>(correct) / static_cast<double>(predictions.rows());
}

inline double r2_score(const Matrix& predictions, const Matrix& targets) {
    double target_mean = 0.0;
    for (std::size_t i = 0; i < targets.rows(); ++i) {
        target_mean += targets(i, 0);
    }
    target_mean /= static_cast<double>(targets.rows());
    double ss_tot = 0.0;
    double ss_res = 0.0;
    for (std::size_t i = 0; i < targets.rows(); ++i) {
        const double diff_mean = targets(i, 0) - target_mean;
        const double diff_pred = targets(i, 0) - predictions(i, 0);
        ss_tot += diff_mean * diff_mean;
        ss_res += diff_pred * diff_pred;
    }
    return 1.0 - (ss_res / ss_tot);
}

}  // namespace ml
