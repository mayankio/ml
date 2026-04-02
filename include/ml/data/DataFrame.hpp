#pragma once

#include <map>
#include <string>
#include <vector>

#include "ml/core/Matrix.hpp"

namespace ml {

class DataFrame {
public:
    DataFrame() = default;
    explicit DataFrame(std::vector<std::string> columns);

    void add_row(const std::vector<std::string>& row);
    [[nodiscard]] std::size_t rows() const;
    [[nodiscard]] std::size_t cols() const;
    [[nodiscard]] const std::vector<std::string>& columns() const;
    [[nodiscard]] const std::vector<std::vector<std::string>>& data() const;
    [[nodiscard]] std::vector<std::string> column(const std::string& name) const;
    [[nodiscard]] std::size_t column_index(const std::string& name) const;
    [[nodiscard]] Matrix numeric_matrix(const std::vector<std::string>& selected_columns = {}) const;

private:
    std::vector<std::string> columns_;
    std::map<std::string, std::size_t> column_to_index_;
    std::vector<std::vector<std::string>> rows_;
};

}  // namespace ml
