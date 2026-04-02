#include <iostream>
#include <vector>

#include "ml/data/CSVReader.hpp"
#include "ml/data/Split.hpp"
#include "ml/data/Transformers.hpp"
#include "ml/linear/LogisticRegression.hpp"

int main() {
    ml::CSVReader reader(',');
    ml::DataFrame iris = reader.read("tests/data/Iris.csv");

    ml::Matrix x = iris.numeric_matrix({"sepal_length", "sepal_width", "petal_length", "petal_width"});
    std::vector<std::string> species = iris.column("species");

    ml::LabelEncoder label_encoder;
    std::vector<int> ids = label_encoder.fit_transform(species);
    ml::Matrix y(ids.size(), 1);
    for (std::size_t i = 0; i < ids.size(); ++i) {
        y(i, 0) = ids[i] == 0 ? 0.0 : 1.0;
    }

    ml::StandardScaler scaler;
    ml::Matrix x_scaled = scaler.fit_transform(x);
    ml::MatrixSplit split = ml::train_test_split(x_scaled, y, 0.2, 42);

    ml::LogisticRegression model(0.1, 2000);
    model.fit(split.x_train, split.y_train);
    ml::Matrix preds = model.predict(split.x_test);

    std::cout << "Test rows: " << split.x_test.rows() << '\n';
    std::cout << "First prediction: " << preds(0, 0) << '\n';
}
