#include "ml/data/Transformers.hpp"

#include <algorithm>
#include <stdexcept>

namespace ml {

void StandardScaler::fit(const Matrix& data) {
    means_ = data.column_means();
    stds_ = data.column_stds(0.0);
    for (double& value : stds_) {
        if (value == 0.0) {
            value = 1.0;
        }
    }
}

Matrix StandardScaler::transform(const Matrix& data) const {
    if (means_.empty() || stds_.empty()) {
        throw std::logic_error("StandardScaler must be fit before transform");
    }
    Matrix output(data.rows(), data.cols());
    for (std::size_t i = 0; i < data.rows(); ++i) {
        for (std::size_t j = 0; j < data.cols(); ++j) {
            output(i, j) = (data(i, j) - means_[j]) / stds_[j];
        }
    }
    return output;
}

const std::vector<double>& StandardScaler::means() const {
    return means_;
}

const std::vector<double>& StandardScaler::stds() const {
    return stds_;
}

MinMaxScaler::MinMaxScaler(double feature_min, double feature_max)
    : feature_min_(feature_min), feature_max_(feature_max) {}

void MinMaxScaler::fit(const Matrix& data) {
    mins_.assign(data.cols(), 0.0);
    maxs_.assign(data.cols(), 0.0);
    for (std::size_t j = 0; j < data.cols(); ++j) {
        mins_[j] = data(0, j);
        maxs_[j] = data(0, j);
        for (std::size_t i = 1; i < data.rows(); ++i) {
            mins_[j] = std::min(mins_[j], data(i, j));
            maxs_[j] = std::max(maxs_[j], data(i, j));
        }
    }
}

Matrix MinMaxScaler::transform(const Matrix& data) const {
    if (mins_.empty() || maxs_.empty()) {
        throw std::logic_error("MinMaxScaler must be fit before transform");
    }
    Matrix output(data.rows(), data.cols());
    for (std::size_t i = 0; i < data.rows(); ++i) {
        for (std::size_t j = 0; j < data.cols(); ++j) {
            const double range = maxs_[j] - mins_[j];
            if (range == 0.0) {
                output(i, j) = feature_min_;
            } else {
                const double normalized = (data(i, j) - mins_[j]) / range;
                output(i, j) = feature_min_ + normalized * (feature_max_ - feature_min_);
            }
        }
    }
    return output;
}

const std::vector<double>& MinMaxScaler::mins() const {
    return mins_;
}

const std::vector<double>& MinMaxScaler::maxs() const {
    return maxs_;
}

void OneHotEncoder::fit(const std::vector<std::string>& values) {
    categories_.clear();
    index_.clear();
    for (const std::string& value : values) {
        if (index_.find(value) == index_.end()) {
            index_[value] = categories_.size();
            categories_.push_back(value);
        }
    }
}

Matrix OneHotEncoder::transform(const std::vector<std::string>& values) const {
    if (categories_.empty()) {
        throw std::logic_error("OneHotEncoder must be fit before transform");
    }
    Matrix output(values.size(), categories_.size(), 0.0);
    for (std::size_t i = 0; i < values.size(); ++i) {
        const auto it = index_.find(values[i]);
        if (it == index_.end()) {
            throw std::invalid_argument("unknown category: " + values[i]);
        }
        output(i, it->second) = 1.0;
    }
    return output;
}

const std::vector<std::string>& OneHotEncoder::categories() const {
    return categories_;
}

void LabelEncoder::fit(const std::vector<std::string>& values) {
    classes_.clear();
    to_id_.clear();
    for (const std::string& value : values) {
        if (to_id_.find(value) == to_id_.end()) {
            const int next = static_cast<int>(classes_.size());
            to_id_[value] = next;
            classes_.push_back(value);
        }
    }
}

std::vector<int> LabelEncoder::transform(const std::vector<std::string>& values) const {
    if (classes_.empty()) {
        throw std::logic_error("LabelEncoder must be fit before transform");
    }
    std::vector<int> output(values.size(), 0);
    for (std::size_t i = 0; i < values.size(); ++i) {
        const auto it = to_id_.find(values[i]);
        if (it == to_id_.end()) {
            throw std::invalid_argument("unknown label: " + values[i]);
        }
        output[i] = it->second;
    }
    return output;
}

std::vector<std::string> LabelEncoder::inverse_transform(const std::vector<int>& values) const {
    std::vector<std::string> output(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (values[i] < 0 || static_cast<std::size_t>(values[i]) >= classes_.size()) {
            throw std::invalid_argument("label id out of range");
        }
        output[i] = classes_[values[i]];
    }
    return output;
}

const std::vector<std::string>& LabelEncoder::classes() const {
    return classes_;
}

}  // namespace ml
