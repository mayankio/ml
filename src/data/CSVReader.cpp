#include "ml/data/CSVReader.hpp"

#include <fstream>
#include <sstream>
#include <stdexcept>

namespace ml {

CSVReader::CSVReader(char delimiter) : delimiter_(delimiter) {}

DataFrame CSVReader::read(const std::string& path, bool has_header) const {
    std::ifstream file(path);
    if (!file) {
        throw std::runtime_error("could not open csv file: " + path);
    }

    std::string line;
    std::vector<std::string> columns;
    bool first_row = true;
    DataFrame frame;
    std::size_t expected_width = 0;

    while (std::getline(file, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        if (line.empty()) {
            continue;
        }
        std::vector<std::string> parsed = parse_line(line);
        if (first_row && has_header) {
            columns = parsed;
            frame = DataFrame(columns);
            expected_width = columns.size();
            first_row = false;
            continue;
        }
        if (first_row && !has_header) {
            columns.resize(parsed.size());
            for (std::size_t i = 0; i < parsed.size(); ++i) {
                std::ostringstream name;
                name << "column_" << i;
                columns[i] = name.str();
            }
            frame = DataFrame(columns);
            expected_width = columns.size();
            first_row = false;
        }
        if (parsed.size() != expected_width) {
            throw std::invalid_argument("inconsistent csv row width");
        }
        for (std::string& cell : parsed) {
            cell = normalize_missing(cell);
        }
        frame.add_row(parsed);
    }

    return frame;
}

char CSVReader::delimiter() const {
    return delimiter_;
}

std::vector<std::string> CSVReader::parse_line(const std::string& line) const {
    std::vector<std::string> values;
    std::string current;
    bool in_quotes = false;
    for (std::size_t i = 0; i < line.size(); ++i) {
        const char c = line[i];
        if (c == '"') {
            if (in_quotes && i + 1 < line.size() && line[i + 1] == '"') {
                current.push_back('"');
                ++i;
            } else {
                in_quotes = !in_quotes;
            }
            continue;
        }
        if (c == delimiter_ && !in_quotes) {
            values.push_back(current);
            current.clear();
            continue;
        }
        current.push_back(c);
    }
    values.push_back(current);
    return values;
}

std::string CSVReader::normalize_missing(const std::string& value) const {
    if (value.empty() || value == "NA" || value == "N/A" || value == "null") {
        return "NaN";
    }
    return value;
}

}  // namespace ml
