#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <functional>
#include <initializer_list>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace ml {

class Matrix {
public:
    Matrix() = default;
    Matrix(std::size_t rows, std::size_t cols, double value = 0.0)
        : rows_(rows), cols_(cols), data_(rows * cols, value) {}

    Matrix(std::initializer_list<std::initializer_list<double>> values) {
        rows_ = values.size();
        cols_ = values.begin()->size();
        data_.reserve(rows_ * cols_);
        for (const auto& row : values) {
            if (row.size() != cols_) {
                throw std::invalid_argument("ragged matrix literal");
            }
            data_.insert(data_.end(), row.begin(), row.end());
        }
    }

    [[nodiscard]] std::size_t rows() const { return rows_; }
    [[nodiscard]] std::size_t cols() const { return cols_; }
    [[nodiscard]] bool empty() const { return data_.empty(); }

    double& operator()(std::size_t row, std::size_t col) {
        return data_.at(row * cols_ + col);
    }

    double operator()(std::size_t row, std::size_t col) const {
        return data_.at(row * cols_ + col);
    }

    [[nodiscard]] const std::vector<double>& data() const { return data_; }
    [[nodiscard]] std::vector<double>& data() { return data_; }

    [[nodiscard]] Matrix row(std::size_t index) const {
        Matrix result(1, cols_);
        for (std::size_t j = 0; j < cols_; ++j) {
            result(0, j) = (*this)(index, j);
        }
        return result;
    }

    [[nodiscard]] Matrix col(std::size_t index) const {
        Matrix result(rows_, 1);
        for (std::size_t i = 0; i < rows_; ++i) {
            result(i, 0) = (*this)(i, index);
        }
        return result;
    }

    [[nodiscard]] std::vector<double> row_vector(std::size_t index) const {
        std::vector<double> result(cols_);
        for (std::size_t j = 0; j < cols_; ++j) {
            result[j] = (*this)(index, j);
        }
        return result;
    }

    [[nodiscard]] Matrix transpose() const {
        Matrix result(cols_, rows_);
        for (std::size_t i = 0; i < rows_; ++i) {
            for (std::size_t j = 0; j < cols_; ++j) {
                result(j, i) = (*this)(i, j);
            }
        }
        return result;
    }

    [[nodiscard]] Matrix apply(const std::function<double(double)>& fn) const {
        Matrix result(rows_, cols_);
        for (std::size_t i = 0; i < data_.size(); ++i) {
            result.data_[i] = fn(data_[i]);
        }
        return result;
    }

    [[nodiscard]] std::vector<double> column_means() const {
        std::vector<double> means(cols_, 0.0);
        for (std::size_t j = 0; j < cols_; ++j) {
            for (std::size_t i = 0; i < rows_; ++i) {
                means[j] += (*this)(i, j);
            }
            means[j] /= static_cast<double>(rows_);
        }
        return means;
    }

    [[nodiscard]] std::vector<double> column_stds(double epsilon = 1e-8) const {
        const auto means = column_means();
        std::vector<double> stds(cols_, 0.0);
        for (std::size_t j = 0; j < cols_; ++j) {
            for (std::size_t i = 0; i < rows_; ++i) {
                const double diff = (*this)(i, j) - means[j];
                stds[j] += diff * diff;
            }
            stds[j] = std::sqrt(stds[j] / static_cast<double>(rows_)) + epsilon;
        }
        return stds;
    }

    static Matrix zeros(std::size_t rows, std::size_t cols) {
        return Matrix(rows, cols, 0.0);
    }

    static Matrix identity(std::size_t size) {
        Matrix result(size, size, 0.0);
        for (std::size_t i = 0; i < size; ++i) {
            result(i, i) = 1.0;
        }
        return result;
    }

    static Matrix random(std::size_t rows, std::size_t cols, double min = -0.1, double max = 0.1, std::uint32_t seed = 42) {
        Matrix result(rows, cols);
        std::mt19937 rng(seed);
        std::uniform_real_distribution<double> dist(min, max);
        for (double& value : result.data_) {
            value = dist(rng);
        }
        return result;
    }

private:
    std::size_t rows_ = 0;
    std::size_t cols_ = 0;
    std::vector<double> data_;
};

inline Matrix operator+(const Matrix& lhs, const Matrix& rhs) {
    if (lhs.rows() != rhs.rows() || lhs.cols() != rhs.cols()) {
        throw std::invalid_argument("matrix shape mismatch for addition");
    }
    Matrix result(lhs.rows(), lhs.cols());
    for (std::size_t i = 0; i < lhs.data().size(); ++i) {
        result.data()[i] = lhs.data()[i] + rhs.data()[i];
    }
    return result;
}

inline Matrix operator-(const Matrix& lhs, const Matrix& rhs) {
    if (lhs.rows() != rhs.rows() || lhs.cols() != rhs.cols()) {
        throw std::invalid_argument("matrix shape mismatch for subtraction");
    }
    Matrix result(lhs.rows(), lhs.cols());
    for (std::size_t i = 0; i < lhs.data().size(); ++i) {
        result.data()[i] = lhs.data()[i] - rhs.data()[i];
    }
    return result;
}

inline Matrix operator*(const Matrix& lhs, double scalar) {
    Matrix result(lhs.rows(), lhs.cols());
    for (std::size_t i = 0; i < lhs.data().size(); ++i) {
        result.data()[i] = lhs.data()[i] * scalar;
    }
    return result;
}

inline Matrix operator*(double scalar, const Matrix& rhs) {
    return rhs * scalar;
}

inline Matrix operator/(const Matrix& lhs, double scalar) {
    Matrix result(lhs.rows(), lhs.cols());
    for (std::size_t i = 0; i < lhs.data().size(); ++i) {
        result.data()[i] = lhs.data()[i] / scalar;
    }
    return result;
}

inline Matrix hadamard(const Matrix& lhs, const Matrix& rhs) {
    if (lhs.rows() != rhs.rows() || lhs.cols() != rhs.cols()) {
        throw std::invalid_argument("matrix shape mismatch for hadamard product");
    }
    Matrix result(lhs.rows(), lhs.cols());
    for (std::size_t i = 0; i < lhs.data().size(); ++i) {
        result.data()[i] = lhs.data()[i] * rhs.data()[i];
    }
    return result;
}

inline Matrix matmul(const Matrix& lhs, const Matrix& rhs) {
    if (lhs.cols() != rhs.rows()) {
        throw std::invalid_argument("matrix shape mismatch for multiplication");
    }
    Matrix result(lhs.rows(), rhs.cols(), 0.0);
    for (std::size_t i = 0; i < lhs.rows(); ++i) {
        for (std::size_t k = 0; k < lhs.cols(); ++k) {
            const double value = lhs(i, k);
            for (std::size_t j = 0; j < rhs.cols(); ++j) {
                result(i, j) += value * rhs(k, j);
            }
        }
    }
    return result;
}

inline Matrix add_row_vector(const Matrix& matrix, const std::vector<double>& bias) {
    if (matrix.cols() != bias.size()) {
        throw std::invalid_argument("bias shape mismatch");
    }
    Matrix result = matrix;
    for (std::size_t i = 0; i < matrix.rows(); ++i) {
        for (std::size_t j = 0; j < matrix.cols(); ++j) {
            result(i, j) += bias[j];
        }
    }
    return result;
}

inline double dot(const std::vector<double>& lhs, const std::vector<double>& rhs) {
    if (lhs.size() != rhs.size()) {
        throw std::invalid_argument("vector size mismatch");
    }
    double result = 0.0;
    for (std::size_t i = 0; i < lhs.size(); ++i) {
        result += lhs[i] * rhs[i];
    }
    return result;
}

inline double euclidean_distance(const std::vector<double>& lhs, const std::vector<double>& rhs) {
    if (lhs.size() != rhs.size()) {
        throw std::invalid_argument("vector size mismatch");
    }
    double sum = 0.0;
    for (std::size_t i = 0; i < lhs.size(); ++i) {
        const double diff = lhs[i] - rhs[i];
        sum += diff * diff;
    }
    return std::sqrt(sum);
}

inline double mean_squared_error(const Matrix& predictions, const Matrix& targets) {
    if (predictions.rows() != targets.rows() || predictions.cols() != targets.cols()) {
        throw std::invalid_argument("matrix shape mismatch for mse");
    }
    double sum = 0.0;
    for (std::size_t i = 0; i < predictions.data().size(); ++i) {
        const double diff = predictions.data()[i] - targets.data()[i];
        sum += diff * diff;
    }
    return sum / static_cast<double>(predictions.data().size());
}

inline double sigmoid(double value) {
    return 1.0 / (1.0 + std::exp(-value));
}

inline double relu(double value) {
    return std::max(0.0, value);
}

inline double relu_derivative(double value) {
    return value > 0.0 ? 1.0 : 0.0;
}

inline double softmax_denominator(const std::vector<double>& logits) {
    const double max_logit = *std::max_element(logits.begin(), logits.end());
    double sum = 0.0;
    for (double value : logits) {
        sum += std::exp(value - max_logit);
    }
    return sum;
}

inline std::vector<double> softmax(const std::vector<double>& logits) {
    const double max_logit = *std::max_element(logits.begin(), logits.end());
    std::vector<double> result(logits.size());
    double sum = 0.0;
    for (std::size_t i = 0; i < logits.size(); ++i) {
        result[i] = std::exp(logits[i] - max_logit);
        sum += result[i];
    }
    for (double& value : result) {
        value /= sum;
    }
    return result;
}

inline std::size_t argmax(const std::vector<double>& values) {
    return static_cast<std::size_t>(std::distance(values.begin(), std::max_element(values.begin(), values.end())));
}

inline void save_matrix(std::ostream& out, const Matrix& matrix) {
    out << matrix.rows() << ' ' << matrix.cols() << '\n';
    for (double value : matrix.data()) {
        out << value << ' ';
    }
    out << '\n';
}

inline Matrix load_matrix(std::istream& in) {
    std::size_t rows = 0;
    std::size_t cols = 0;
    in >> rows >> cols;
    Matrix matrix(rows, cols);
    for (double& value : matrix.data()) {
        in >> value;
    }
    return matrix;
}

template <typename T>
inline void save_vector(std::ostream& out, const std::vector<T>& values) {
    out << values.size() << '\n';
    for (const T& value : values) {
        out << value << ' ';
    }
    out << '\n';
}

template <typename T>
inline std::vector<T> load_vector(std::istream& in) {
    std::size_t size = 0;
    in >> size;
    std::vector<T> values(size);
    for (T& value : values) {
        in >> value;
    }
    return values;
}

}  // namespace ml
