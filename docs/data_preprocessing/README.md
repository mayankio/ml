# Data Plumbing and Preprocessing

This module adds a lightweight data-ingestion pipeline for the library:

- `ml::CSVReader` parses CSV files into `ml::DataFrame`
- `ml::StandardScaler` and `ml::MinMaxScaler` normalize numeric matrices
- `ml::OneHotEncoder` and `ml::LabelEncoder` convert strings into numeric representations
- `ml::train_test_split` shuffles rows and returns train/test matrices

## Typical Pipeline

```cpp
#include "ml/data/CSVReader.hpp"
#include "ml/data/Split.hpp"
#include "ml/data/Transformers.hpp"
#include "ml/linear/LogisticRegression.hpp"

ml::CSVReader reader(',');
ml::DataFrame iris = reader.read("tests/data/Iris.csv");

ml::Matrix x = iris.numeric_matrix({
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width"
});

ml::LabelEncoder label_encoder;
std::vector<int> y_ids = label_encoder.fit_transform(iris.column("species"));
ml::Matrix y(y_ids.size(), 1);
for (std::size_t i = 0; i < y_ids.size(); ++i) {
    y(i, 0) = static_cast<double>(y_ids[i] == 0 ? 0 : 1);
}

ml::StandardScaler scaler;
ml::Matrix x_scaled = scaler.fit_transform(x);

ml::MatrixSplit split = ml::train_test_split(x_scaled, y, 0.2, 42);

ml::LogisticRegression model(0.1, 2000);
model.fit(split.x_train, split.y_train);
ml::Matrix preds = model.predict(split.x_test);
```

## CSV Parsing Notes

- Quoted strings are supported, including delimiters inside quotes.
- Missing values such as empty strings, `NA`, `N/A`, and `null` are normalized to the string `"NaN"`.
- Inconsistent row lengths raise an exception.

## Scaling Math

### StandardScaler

\[
z = \frac{x - \mu}{\sigma}
\]

If a feature has zero variance, this implementation uses `sigma = 1` to avoid division by zero.

### MinMaxScaler

\[
x' = a + \frac{(x - x_{min})(b-a)}{x_{max} - x_{min}}
\]

For the default range, `a = 0` and `b = 1`.

## Encoders

- `LabelEncoder` maps strings like `"Setosa"` to integer ids and can invert them back to strings.
- `OneHotEncoder` maps each category to a one-of-K binary row.

## Tests

The parser and preprocessors are exercised in [test_data_module.cpp](/Users/dksingh/src/ml/tests/test_data_module.cpp), including:

- row-count verification for a 150-row Iris fixture
- quoted-string parsing
- missing-value normalization
- exact scaling checks for standardization and min-max scaling
