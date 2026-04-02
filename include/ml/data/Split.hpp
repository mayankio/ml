#pragma once

#include <string>
#include <vector>

#include "ml/core/Matrix.hpp"

namespace ml {

struct MatrixSplit {
    Matrix x_train;
    Matrix x_test;
    Matrix y_train;
    Matrix y_test;
};

MatrixSplit train_test_split(const Matrix& features, const Matrix& targets, double test_ratio = 0.2, unsigned int seed = 42);

}  // namespace ml
