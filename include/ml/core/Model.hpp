#pragma once

#include <string>

#include "ml/core/Matrix.hpp"

namespace ml {

class Model {
public:
    virtual ~Model() = default;
    virtual void fit(const Matrix& features, const Matrix& targets) = 0;
    virtual Matrix predict(const Matrix& features) const = 0;
    virtual void save(const std::string& path) const = 0;
    virtual void load(const std::string& path) = 0;
};

}  // namespace ml
