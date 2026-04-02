#include "ml/data/DataFrame.hpp"

#include <stdexcept>
#include <utility>

namespace ml {

DataFrame::DataFrame(std::vector<std::string> columns) : columns_(std::move(columns)) {
    for (std::size_t i = 0; i < columns_.size(); ++i) {
        column_to_index_[columns_[i]] = i;
    }
}

void DataFrame::add_row(const std::vector<std::string>& row) {
    if (!columns_.empty() && row.size() != columns_.size()) {
        throw std::invalid_argument("row width does not match dataframe columns");
    }
    rows_.push_back(row);
}

std::size_t DataFrame::rows() const {
    return rows_.size();
}

std::size_t DataFrame::cols() const {
    return columns_.size();
}

const std::vector<std::string>& DataFrame::columns() const {
    return columns_;
}

const std::vector<std::vector<std::string>>& DataFrame::data() const {
    return rows_;
}

std::vector<std::string> DataFrame::column(const std::string& name) const {
    const std::size_t index = column_index(name);
    std::vector<std::string> values(rows_.size(), "");
    for (std::size_t i = 0; i < rows_.size(); ++i) {
        values[i] = rows_[i][index];
    }
    return values;
}

std::size_t DataFrame::column_index(const std::string& name) const {
    const auto it = column_to_index_.find(name);
    if (it == column_to_index_.end()) {
        throw std::invalid_argument("unknown column: " + name);
    }
    return it->second;
}

Matrix DataFrame::numeric_matrix(const std::vector<std::string>& selected_columns) const {
    std::vector<std::size_t> indices;
    if (selected_columns.empty()) {
        indices.resize(columns_.size());
        for (std::size_t i = 0; i < columns_.size(); ++i) {
            indices[i] = i;
        }
    } else {
        for (const auto& name : selected_columns) {
            indices.push_back(column_index(name));
        }
    }
    Matrix matrix(rows_.size(), indices.size());
    for (std::size_t i = 0; i < rows_.size(); ++i) {
        for (std::size_t j = 0; j < indices.size(); ++j) {
            const std::string& cell = rows_[i][indices[j]];
            if (cell == "NaN" || cell.empty()) {
                matrix(i, j) = 0.0;
            } else {
                matrix(i, j) = std::stod(cell);
            }
        }
    }
    return matrix;
}

}  // namespace ml
