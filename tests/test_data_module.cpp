#include <cassert>

#include "TestUtils.hpp"
#include "ml/data/CSVReader.hpp"
#include "ml/data/Split.hpp"
#include "ml/data/Transformers.hpp"

int main() {
    {
        ml::CSVReader reader(',');
        const ml::DataFrame iris = reader.read("tests/data/Iris.csv");
        assert(iris.rows() == 150);
        assert(iris.cols() == 5);
        const ml::Matrix iris_x = iris.numeric_matrix({"sepal_length", "sepal_width", "petal_length", "petal_width"});
        assert(iris_x.rows() == 150);
        assert(iris_x.cols() == 4);
    }

    {
        ml::CSVReader reader(',');
        const ml::DataFrame mixed = reader.read("tests/data/mixed.csv");
        assert(mixed.rows() == 3);
        assert(mixed.column("name")[0] == "Ada Lovelace");
        assert(mixed.column("city")[0] == "London, UK");
        assert(mixed.column("age")[1] == "NaN");
        assert(mixed.column("score")[2] == "NaN");
    }

    {
        ml::Matrix x{{1.0, 10.0}, {2.0, 20.0}, {3.0, 30.0}};
        ml::StandardScaler scaler;
        const ml::Matrix scaled = scaler.fit_transform(x);
        assert_close(scaled(0, 0), -1.224744871, 1e-6);
        assert_close(scaled(1, 0), 0.0, 1e-9);
        assert_close(scaled(2, 0), 1.224744871, 1e-6);
        assert_close(scaled(0, 1), -1.224744871, 1e-6);
    }

    {
        ml::Matrix x{{2.0, 10.0}, {4.0, 20.0}, {6.0, 30.0}};
        ml::MinMaxScaler scaler;
        const ml::Matrix scaled = scaler.fit_transform(x);
        assert_close(scaled(0, 0), 0.0, 1e-9);
        assert_close(scaled(1, 0), 0.5, 1e-9);
        assert_close(scaled(2, 0), 1.0, 1e-9);
        assert_close(scaled(1, 1), 0.5, 1e-9);
    }

    {
        std::vector<std::string> labels{"Setosa", "Versicolor", "Virginica", "Setosa"};
        ml::LabelEncoder encoder;
        const std::vector<int> encoded = encoder.fit_transform(labels);
        assert(encoded[0] == 0);
        assert(encoded[1] == 1);
        assert(encoded[2] == 2);
        const std::vector<std::string> decoded = encoder.inverse_transform(encoded);
        assert(decoded == labels);
    }

    {
        std::vector<std::string> colors{"red", "blue", "red", "green"};
        ml::OneHotEncoder encoder;
        const ml::Matrix encoded = encoder.fit_transform(colors);
        assert(encoded.rows() == 4);
        assert(encoded.cols() == 3);
        assert(encoded(0, 0) + encoded(0, 1) + encoded(0, 2) == 1.0);
        assert(encoded(1, 0) + encoded(1, 1) + encoded(1, 2) == 1.0);
    }

    {
        ml::Matrix x{{1.0}, {2.0}, {3.0}, {4.0}, {5.0}};
        ml::Matrix y{{0.0}, {0.0}, {1.0}, {1.0}, {1.0}};
        const ml::MatrixSplit split = ml::train_test_split(x, y, 0.4, 7);
        assert(split.x_train.rows() == 3);
        assert(split.x_test.rows() == 2);
        assert(split.y_train.rows() == 3);
        assert(split.y_test.rows() == 2);
    }
}
