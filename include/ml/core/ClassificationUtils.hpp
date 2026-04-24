#pragma once

#include <cmath>
#include <map>
#include <set>
#include <stdexcept>
#include <vector>

#include "ml/core/Matrix.hpp"

namespace ml {

struct ClassificationTargetInfo {
    std::vector<int> classes;
    std::vector<std::size_t> indices;
};

inline int checked_class_label(double value) {
    if (!std::isfinite(value)) {
        throw std::invalid_argument("classification targets must be finite");
    }
    const long long rounded = std::llround(value);
    if (std::fabs(value - static_cast<double>(rounded)) > 1e-9) {
        throw std::invalid_argument("classification targets must be integer-valued");
    }
    return static_cast<int>(rounded);
}

inline ClassificationTargetInfo parse_classification_targets(const Matrix& features, const Matrix& targets) {
    if (features.rows() == 0 || features.cols() == 0) {
        throw std::invalid_argument("classification features must be non-empty");
    }
    if (features.rows() != targets.rows()) {
        throw std::invalid_argument("feature and target row counts must match");
    }
    if (targets.cols() != 1) {
        throw std::invalid_argument("classification targets must have exactly one column");
    }

    std::vector<int> labels(targets.rows(), 0);
    std::set<int> class_set;
    for (std::size_t i = 0; i < targets.rows(); ++i) {
        labels[i] = checked_class_label(targets(i, 0));
        class_set.insert(labels[i]);
    }
    if (class_set.size() < 2) {
        throw std::invalid_argument("classification requires at least two distinct classes");
    }

    ClassificationTargetInfo info;
    info.classes.assign(class_set.begin(), class_set.end());
    std::map<int, std::size_t> class_to_index;
    for (std::size_t i = 0; i < info.classes.size(); ++i) {
        class_to_index[info.classes[i]] = i;
    }
    info.indices.resize(labels.size());
    for (std::size_t i = 0; i < labels.size(); ++i) {
        info.indices[i] = class_to_index.at(labels[i]);
    }
    return info;
}

inline std::size_t output_dimension_for_classes(const std::vector<int>& classes) {
    return classes.size() > 2 ? classes.size() : 1;
}

inline bool is_binary_classes(const std::vector<int>& classes) {
    return classes.size() == 2;
}

inline Matrix decode_binary_predictions(const std::vector<int>& classes, const Matrix& scores_or_probs, double threshold) {
    if (classes.size() != 2) {
        throw std::invalid_argument("binary prediction decoding requires exactly two classes");
    }
    if (scores_or_probs.cols() != 1) {
        throw std::invalid_argument("binary prediction decoding expects a single output column");
    }
    Matrix output(scores_or_probs.rows(), 1);
    for (std::size_t i = 0; i < scores_or_probs.rows(); ++i) {
        output(i, 0) = scores_or_probs(i, 0) >= threshold ? static_cast<double>(classes[1]) : static_cast<double>(classes[0]);
    }
    return output;
}

inline Matrix decode_multiclass_predictions(const std::vector<int>& classes, const Matrix& scores) {
    if (classes.size() < 3) {
        throw std::invalid_argument("multiclass prediction decoding requires at least three classes");
    }
    if (scores.cols() != classes.size()) {
        throw std::invalid_argument("score matrix column count must match class count");
    }
    Matrix output(scores.rows(), 1);
    for (std::size_t i = 0; i < scores.rows(); ++i) {
        output(i, 0) = static_cast<double>(classes[argmax(scores.row_vector(i))]);
    }
    return output;
}

inline Matrix build_probability_matrix(const std::vector<std::vector<double>>& rows) {
    if (rows.empty()) {
        return Matrix{};
    }
    Matrix output(rows.size(), rows.front().size());
    for (std::size_t i = 0; i < rows.size(); ++i) {
        if (rows[i].size() != output.cols()) {
            throw std::invalid_argument("probability rows must share the same width");
        }
        for (std::size_t j = 0; j < rows[i].size(); ++j) {
            output(i, j) = rows[i][j];
        }
    }
    return output;
}

}  // namespace ml
