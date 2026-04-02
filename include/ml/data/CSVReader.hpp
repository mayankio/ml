#pragma once

#include <string>
#include <vector>

#include "ml/data/DataFrame.hpp"

namespace ml {

class CSVReader {
public:
    explicit CSVReader(char delimiter = ',');

    DataFrame read(const std::string& path, bool has_header = true) const;
    [[nodiscard]] char delimiter() const;

private:
    std::vector<std::string> parse_line(const std::string& line) const;
    std::string normalize_missing(const std::string& value) const;

    char delimiter_;
};

}  // namespace ml
