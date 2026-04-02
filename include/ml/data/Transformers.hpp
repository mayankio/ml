#pragma once

#include <map>
#include <string>
#include <vector>

#include "ml/core/Matrix.hpp"

namespace ml {

class MatrixTransformer {
public:
    virtual ~MatrixTransformer() = default;
    virtual void fit(const Matrix& data) = 0;
    virtual Matrix transform(const Matrix& data) const = 0;
    virtual Matrix fit_transform(const Matrix& data) {
        fit(data);
        return transform(data);
    }
};

class StandardScaler : public MatrixTransformer {
public:
    void fit(const Matrix& data) override;
    Matrix transform(const Matrix& data) const override;
    [[nodiscard]] const std::vector<double>& means() const;
    [[nodiscard]] const std::vector<double>& stds() const;

private:
    std::vector<double> means_;
    std::vector<double> stds_;
};

class MinMaxScaler : public MatrixTransformer {
public:
    MinMaxScaler(double feature_min = 0.0, double feature_max = 1.0);

    void fit(const Matrix& data) override;
    Matrix transform(const Matrix& data) const override;
    [[nodiscard]] const std::vector<double>& mins() const;
    [[nodiscard]] const std::vector<double>& maxs() const;

private:
    double feature_min_;
    double feature_max_;
    std::vector<double> mins_;
    std::vector<double> maxs_;
};

class OneHotEncoder {
public:
    void fit(const std::vector<std::string>& values);
    Matrix transform(const std::vector<std::string>& values) const;
    Matrix fit_transform(const std::vector<std::string>& values) {
        fit(values);
        return transform(values);
    }
    [[nodiscard]] const std::vector<std::string>& categories() const;

private:
    std::vector<std::string> categories_;
    std::map<std::string, std::size_t> index_;
};

class LabelEncoder {
public:
    void fit(const std::vector<std::string>& values);
    std::vector<int> transform(const std::vector<std::string>& values) const;
    std::vector<int> fit_transform(const std::vector<std::string>& values) {
        fit(values);
        return transform(values);
    }
    std::vector<std::string> inverse_transform(const std::vector<int>& values) const;
    [[nodiscard]] const std::vector<std::string>& classes() const;

private:
    std::vector<std::string> classes_;
    std::map<std::string, int> to_id_;
};

}  // namespace ml
